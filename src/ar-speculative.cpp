#include "ar-speculative.h"
#include "ar-graph.h"
#include "diffuse.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

// ── Helpers ─────────────────────────────────────────────────────

static int32_t argmax(const float * logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Qwen2.5 family EOS tokens
static bool is_eos_token(int32_t token) {
    return token == 151643 || token == 151645 || token == 151644;
}

// ── Speculative Generation with Deferred Bonus ──────────────────
//
// Key optimization: instead of decoding the bonus token on the target
// as a separate forward pass (~540ms for bandwidth-bound 32B model),
// we defer it to the NEXT iteration's batch. Each batch is:
//
//   [pending_bonus, draft_d1, draft_d2, ..., draft_dK]   (K+1 tokens)
//
// The causal mask ensures batch_logits[0] depends only on context +
// pending (not on d1..dK), so verification is correct.
//
// This eliminates 1 target forward per iteration:
//   Before: 1 batch + 1 decode = 2 forwards per iteration
//   After:  1 batch = 1 forward per iteration
//
// The target and draft caches stay 1 position apart (draft has the
// pending token decoded, target doesn't until the next batch).

std::vector<int32_t> ar_speculative_generate(
        diffuse_context * target_ctx,
        diffuse_context * draft_ctx,
        const std::vector<int32_t> & prompt_tokens,
        int max_tokens,
        const ar_spec_params & params,
        ar_token_callback callback,
        ar_spec_stats * stats_out) {

    const auto & t_hp = target_ctx->model->hparams;
    const auto & d_hp = draft_ctx->model->hparams;
    const int t_vocab = (int)t_hp.n_vocab;
    const int d_vocab = (int)d_hp.n_vocab;
    const int prompt_len = (int)prompt_tokens.size();
    const int total_ctx = prompt_len + max_tokens + params.K + 4;
    const int K = std::max(params.K, 1);

    DIFFUSE_LOG("speculative: target=%dL/%dH/%dKV vocab=%d, draft=%dL/%dH/%dKV vocab=%d, K=%d",
                (int)t_hp.n_layer, (int)t_hp.n_head, (int)t_hp.n_head_kv, t_vocab,
                (int)d_hp.n_layer, (int)d_hp.n_head, (int)d_hp.n_head_kv, d_vocab, K);

    // ── Init KV caches ──────────────────────────────────────────
    ar_kv_cache target_cache, draft_cache;
    target_cache.init(total_ctx, (int)t_hp.n_layer,
                      (int)t_hp.n_embd_head(), (int)t_hp.n_head_kv);
    draft_cache.init(total_ctx, (int)d_hp.n_layer,
                     (int)d_hp.n_embd_head(), (int)d_hp.n_head_kv);

    // ── Persistent threadpools ──────────────────────────────────
    {
        auto tp = ggml_threadpool_params_default(target_ctx->n_threads);
        tp.strict_cpu = true;
        tp.prio = GGML_SCHED_PRIO_HIGH;
        target_cache.threadpool = ggml_threadpool_new(&tp);
        DIFFUSE_LOG("  target: %d threads, %dL, %d KV heads, %d dim",
                    target_ctx->n_threads, (int)t_hp.n_layer,
                    (int)t_hp.n_head_kv, (int)t_hp.n_embd_head());
    }
    {
        auto tp = ggml_threadpool_params_default(draft_ctx->n_threads);
        tp.strict_cpu = true;
        tp.prio = GGML_SCHED_PRIO_NORMAL;
        draft_cache.threadpool = ggml_threadpool_new(&tp);
        DIFFUSE_LOG("  draft:  %d threads, %dL, %d KV heads, %d dim",
                    draft_ctx->n_threads, (int)d_hp.n_layer,
                    (int)d_hp.n_head_kv, (int)d_hp.n_embd_head());
    }

    ar_spec_stats stats = {};
    using clk = std::chrono::steady_clock;
    auto gen_start = clk::now();

    // Set sliding window on target cache if configured
    target_cache.sliding_window = params.sliding_window;

    // ── Prefill both models ─────────────────────────────────────
    std::vector<float> t_prefill_logits((size_t)prompt_len * t_vocab);
    {
        std::vector<float> d_prefill_logits((size_t)prompt_len * d_vocab);

        auto t0 = clk::now();

        // Target prefill: optionally profile and set layer skip mask
        if (params.layer_skip > 0) {
            if (!ar_profile_layers(target_ctx, prompt_tokens.data(), prompt_len,
                                   &target_cache, t_prefill_logits.data(),
                                   params.layer_skip)) {
                DIFFUSE_DIE("speculative: target profile prefill failed");
            }
        } else {
            if (!ar_forward_prefill(target_ctx, prompt_tokens.data(), prompt_len,
                                    &target_cache, t_prefill_logits.data())) {
                DIFFUSE_DIE("speculative: target prefill failed");
            }
        }

        if (!ar_forward_prefill(draft_ctx, prompt_tokens.data(), prompt_len,
                                &draft_cache, d_prefill_logits.data())) {
            DIFFUSE_DIE("speculative: draft prefill failed");
        }
        auto t1 = clk::now();
        stats.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        DIFFUSE_LOG("  prefill: %d tokens in %.1f ms (%.1f tok/s)",
                    prompt_len, stats.prefill_ms,
                    1000.0 * prompt_len / std::max(stats.prefill_ms, 0.1));
    }

    // ── Sample first token from target prefill logits ───────────
    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    int32_t first_token = argmax(
        t_prefill_logits.data() + (size_t)(prompt_len - 1) * t_vocab, t_vocab);

    // Free prefill buffer
    t_prefill_logits.clear();
    t_prefill_logits.shrink_to_fit();

    generated.push_back(first_token);
    stats.total_generated++;

    if (callback && !callback(first_token, prompt_len)) {
        goto cleanup;
    }
    if (is_eos_token(first_token)) {
        DIFFUSE_LOG("  EOS at first token (%d)", first_token);
        goto cleanup;
    }

    // ── Deferred bonus: DON'T decode first_token on target ──────
    // Instead, set it as pending and include it in the first batch.
    // This saves one target forward (~540ms for 32B).
    //
    // Only decode on draft (fast, ~12ms) to get prev_draft_logits.
    {
        std::vector<float> prev_draft_logits(d_vocab);
        if (!ar_forward_decode(draft_ctx, first_token,
                               &draft_cache, prev_draft_logits.data())) {
            DIFFUSE_DIE("speculative: draft decode first token failed");
        }
        // State:
        //   target_cache.n_past = prompt_len      (first_token NOT decoded)
        //   draft_cache.n_past  = prompt_len + 1  (first_token decoded)
        //   pending = first_token

        int32_t pending = first_token;

        // ── Main speculative loop ───────────────────────────────
        auto decode_start = clk::now();

        std::vector<float> draft_logits_buf(d_vocab);
        std::vector<float> target_batch_logits;
        std::vector<int32_t> candidates;
        std::vector<int32_t> batch_tokens;
        candidates.reserve(K);
        batch_tokens.reserve(K + 1);

        while ((int)generated.size() < max_tokens) {
            int remaining = max_tokens - (int)generated.size();
            int k = std::min(K, remaining);

            int target_P = target_cache.n_past;
            // Invariant: draft_cache.n_past == target_P + 1

            // ── Phase 1: Draft generates k candidates ───────────
            candidates.clear();
            {
                // First candidate from prev_draft_logits
                candidates.push_back(argmax(prev_draft_logits.data(), d_vocab));

                // Remaining: decode on draft and sample
                for (int i = 1; i < k; i++) {
                    if (!ar_forward_decode(draft_ctx, candidates[i - 1],
                                           &draft_cache, draft_logits_buf.data())) {
                        DIFFUSE_LOG("speculative: draft decode failed at candidate %d", i);
                        goto loop_done;
                    }
                    candidates.push_back(argmax(draft_logits_buf.data(), d_vocab));
                }
                stats.total_draft_tokens += k;
            }
            // draft_cache.n_past = (target_P + 1) + (k - 1) = target_P + k

            // ── Phase 2: Target batch [pending, d1..dk] ─────────
            //
            // K+1 tokens total. Logits layout:
            //   batch_logits[0]: after pending    → verify d1
            //   batch_logits[1]: after d1         → verify d2
            //   ...
            //   batch_logits[k]: after dk         → bonus if all accepted
            //
            // Causal mask ensures batch_logits[i] depends only on
            // context + pending + d1..di (not future draft tokens).
            {
                batch_tokens.clear();
                batch_tokens.push_back(pending);
                batch_tokens.insert(batch_tokens.end(),
                                    candidates.begin(), candidates.end());

                int batch_size = 1 + k;
                target_batch_logits.resize((size_t)batch_size * t_vocab);

                if (!ar_forward_batch(target_ctx, batch_tokens.data(), batch_size,
                                      &target_cache, target_batch_logits.data())) {
                    DIFFUSE_LOG("speculative: target batch failed");
                    goto loop_done;
                }
                stats.total_target_batches++;
                // target_cache.n_past = target_P + batch_size = target_P + 1 + k
            }

            // ── Phase 3: Verify candidates ──────────────────────
            {
                int n_accepted = 0;

                // Verify d_i against batch_logits[i-1]
                //   batch_logits[0] = after pending → predicts d1
                //   batch_logits[1] = after d1      → predicts d2
                //   ...
                for (int i = 0; i < k; i++) {
                    float * bl = target_batch_logits.data()
                               + (size_t)i * t_vocab;
                    if (argmax(bl, t_vocab) == candidates[i]) {
                        n_accepted++;
                    } else {
                        break;
                    }
                }

                // Bonus: target's prediction after the last accepted token
                //   n_accepted=0 → bonus from batch_logits[0] (after pending)
                //   n_accepted=J → bonus from batch_logits[J] (after d_J)
                //   n_accepted=k → bonus from batch_logits[k] (after d_k)
                int32_t bonus = argmax(
                    target_batch_logits.data() + (size_t)n_accepted * t_vocab,
                    t_vocab);

                stats.total_accepted += n_accepted;
                if (n_accepted == 0) stats.total_fast_rejects++;

                // Rollback target: keep pending + accepted candidates
                // target was at target_P + 1 + k, we want target_P + 1 + n_accepted
                target_cache.n_past = target_P + 1 + n_accepted;

                // ── Output accepted tokens ──────────────────────
                for (int i = 0; i < n_accepted; i++) {
                    if ((int)generated.size() >= max_tokens) goto loop_done;
                    generated.push_back(candidates[i]);
                    stats.total_generated++;
                    if (callback && !callback(candidates[i],
                            prompt_len + (int)generated.size() - 1)) {
                        goto loop_done;
                    }
                    if (is_eos_token(candidates[i])) {
                        goto loop_done;
                    }
                }

                // ── Output bonus ────────────────────────────────
                if ((int)generated.size() < max_tokens) {
                    generated.push_back(bonus);
                    stats.total_generated++;
                    if (callback && !callback(bonus,
                            prompt_len + (int)generated.size() - 1)) {
                        goto loop_done;
                    }
                    if (is_eos_token(bonus)) {
                        goto loop_done;
                    }
                }

                // ── Sync draft: rollback + decode bonus ─────────
                //
                // draft was at target_P + k (from generating k-1 decodes)
                // target is now at target_P + 1 + n_accepted
                // We need draft at target_P + 1 + n_accepted,
                // then decode bonus → draft at target_P + 2 + n_accepted
                //
                // delta = (target_P + k) - (target_P + 1 + n_accepted)
                //       = k - 1 - n_accepted
                //
                //   k-1-n > 0: rollback draft
                //   k-1-n = 0: nothing
                //   k-1-n < 0: advance (decode candidates[k-1])
                {
                    int draft_current = target_P + k;
                    int draft_target  = target_P + 1 + n_accepted;

                    if (draft_current > draft_target) {
                        draft_cache.n_past = draft_target;
                    } else if (draft_current < draft_target) {
                        // n_accepted == k: decode the last candidate
                        if (!ar_forward_decode(draft_ctx, candidates[k - 1],
                                               &draft_cache,
                                               draft_logits_buf.data())) {
                            DIFFUSE_LOG("speculative: draft decode last candidate failed");
                            goto loop_done;
                        }
                    }

                    // Decode bonus on draft → get prev_draft_logits
                    if (!ar_forward_decode(draft_ctx, bonus,
                                           &draft_cache,
                                           prev_draft_logits.data())) {
                        DIFFUSE_LOG("speculative: draft decode bonus failed");
                        goto loop_done;
                    }
                }

                // Set pending for next iteration (deferred bonus)
                pending = bonus;

                DIFFUSE_LOG("  spec: k=%d accepted=%d bonus=%d (%.0f%%) [deferred]",
                            k, n_accepted, bonus,
                            k > 0 ? 100.0f * n_accepted / k : 0.0f);
            }
        }

    loop_done:
        auto decode_end = clk::now();
        stats.decode_ms = std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
    }

    // ── Report ──────────────────────────────────────────────────
    {
        auto gen_end = clk::now();
        double total_ms = std::chrono::duration<double, std::milli>(
            gen_end - gen_start).count();

        int total_target = stats.total_target_batches + stats.total_target_decodes;

        DIFFUSE_LOG("=== SPECULATIVE GENERATION COMPLETE ===");
        DIFFUSE_LOG("  prefill: %.1f ms (%d tok, %.1f tok/s)",
                    stats.prefill_ms, prompt_len,
                    1000.0 * prompt_len / std::max(stats.prefill_ms, 0.1));
        DIFFUSE_LOG("  decode:  %.1f ms (%d tok, %.2f tok/s)",
                    stats.decode_ms, stats.total_generated,
                    stats.total_generated > 0
                        ? 1000.0 * stats.total_generated / stats.decode_ms
                        : 0.0);
        DIFFUSE_LOG("  acceptance: %d/%d (%.1f%%)",
                    stats.total_accepted, stats.total_draft_tokens,
                    100.0f * stats.acceptance_rate());
        DIFFUSE_LOG("  d1 rejects: %d (no bonus saved — deferred mode)",
                    stats.total_fast_rejects);
        DIFFUSE_LOG("  target calls: %d batches + %d decodes = %d total",
                    stats.total_target_batches, stats.total_target_decodes,
                    total_target);
        DIFFUSE_LOG("  effective: %.2f tok/target_call (deferred bonus)",
                    stats.tokens_per_target_call());
        DIFFUSE_LOG("  total: %.1f ms", total_ms);
    }

cleanup:
    if (stats_out) *stats_out = stats;

    if (target_cache.threadpool) {
        ggml_threadpool_free(target_cache.threadpool);
        target_cache.threadpool = nullptr;
    }
    if (draft_cache.threadpool) {
        ggml_threadpool_free(draft_cache.threadpool);
        draft_cache.threadpool = nullptr;
    }
    target_cache.clear();
    draft_cache.clear();

    return generated;
}
