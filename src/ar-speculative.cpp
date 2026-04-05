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

// ── Speculative Generation ──────────────────────────────────────

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

    // ── Prefill both models ─────────────────────────────────────
    std::vector<float> t_prefill_logits((size_t)prompt_len * t_vocab);
    {
        // We only need the draft prefill to populate its KV cache, not the logits.
        // But ar_forward_prefill requires a logits buffer, so allocate a temp one.
        std::vector<float> d_prefill_logits((size_t)prompt_len * d_vocab);

        auto t0 = clk::now();
        if (!ar_forward_prefill(target_ctx, prompt_tokens.data(), prompt_len,
                                &target_cache, t_prefill_logits.data())) {
            DIFFUSE_DIE("speculative: target prefill failed");
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

    std::vector<float> prev_target_logits(
        t_prefill_logits.data() + (size_t)(prompt_len - 1) * t_vocab,
        t_prefill_logits.data() + (size_t)prompt_len * t_vocab);

    // Free prefill buffer (large: prompt_len * vocab * 4 bytes)
    t_prefill_logits.clear();
    t_prefill_logits.shrink_to_fit();

    int32_t first_token = argmax(prev_target_logits.data(), t_vocab);
    generated.push_back(first_token);
    stats.total_generated++;

    if (callback && !callback(first_token, prompt_len)) {
        goto cleanup;
    }
    if (is_eos_token(first_token)) {
        DIFFUSE_LOG("  EOS at first token (%d)", first_token);
        goto cleanup;
    }

    // ── Decode first token on both models to sync caches ────────
    {
        std::vector<float> t_logits(t_vocab);
        if (!ar_forward_decode(target_ctx, first_token,
                               &target_cache, t_logits.data())) {
            DIFFUSE_DIE("speculative: target decode first token failed");
        }
        prev_target_logits.assign(t_logits.begin(), t_logits.end());
        stats.total_target_decodes++;
    }

    {
        // Persistent buffer for draft logits (reused across iterations)
        std::vector<float> prev_draft_logits(d_vocab);
        if (!ar_forward_decode(draft_ctx, first_token,
                               &draft_cache, prev_draft_logits.data())) {
            DIFFUSE_DIE("speculative: draft decode first token failed");
        }

        // ── Main speculative loop ───────────────────────────────
        auto decode_start = clk::now();

        std::vector<float> draft_logits_buf(d_vocab);
        std::vector<float> target_batch_logits;
        std::vector<int32_t> candidates;
        candidates.reserve(K);

        while ((int)generated.size() < max_tokens) {
            int remaining = max_tokens - (int)generated.size();
            int k = std::min(K, remaining);

            int target_P = target_cache.n_past;
            int draft_P  = draft_cache.n_past;

            // ── Phase 1: Draft generates k candidates ───────────
            candidates.clear();
            {
                // First candidate from prev_draft_logits (no decode needed)
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
            // draft_cache.n_past = draft_P + (k - 1)
            // Positions draft_P .. draft_P+k-2 have KV for candidates[0..k-2]

            // ── Phase 2: Target verification ────────────────────
            {
                int n_accepted = 0;
                int32_t bonus;

                int32_t target_predicted = argmax(prev_target_logits.data(), t_vocab);

                if (target_predicted != candidates[0]) {
                    // ── Fast path: d1 rejected, skip expensive batch ──
                    bonus = target_predicted;
                    n_accepted = 0;
                    stats.total_fast_rejects++;
                } else {
                    // d1 accepted — batch verify all k candidates
                    target_batch_logits.resize((size_t)k * t_vocab);
                    if (!ar_forward_batch(target_ctx, candidates.data(), k,
                                          &target_cache, target_batch_logits.data())) {
                        DIFFUSE_LOG("speculative: target batch failed");
                        goto loop_done;
                    }
                    stats.total_target_batches++;
                    // target_cache.n_past = target_P + k

                    n_accepted = 1;
                    for (int i = 1; i < k; i++) {
                        float * bl = target_batch_logits.data()
                                   + (size_t)(i - 1) * t_vocab;
                        if (argmax(bl, t_vocab) == candidates[i]) {
                            n_accepted++;
                        } else {
                            break;
                        }
                    }

                    // Determine bonus from target logits
                    if (n_accepted < k) {
                        bonus = argmax(
                            target_batch_logits.data()
                                + (size_t)(n_accepted - 1) * t_vocab,
                            t_vocab);
                    } else {
                        bonus = argmax(
                            target_batch_logits.data()
                                + (size_t)(k - 1) * t_vocab,
                            t_vocab);
                    }

                    // Rollback target cache to accepted prefix
                    target_cache.n_past = target_P + n_accepted;
                }

                stats.total_accepted += n_accepted;

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

                // ── Output bonus token ──────────────────────────
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

                // ── Sync target: decode bonus ───────────────────
                {
                    std::vector<float> t_logits(t_vocab);
                    if (!ar_forward_decode(target_ctx, bonus,
                                           &target_cache, t_logits.data())) {
                        DIFFUSE_LOG("speculative: target decode bonus failed");
                        goto loop_done;
                    }
                    prev_target_logits.assign(t_logits.begin(), t_logits.end());
                    stats.total_target_decodes++;
                }

                // ── Sync draft: rollback + decode bonus ─────────
                //
                // Draft decoded k-1 times → n_past = draft_P + k - 1
                // We need n_past = draft_P + n_accepted, then decode bonus.
                //
                // Cases:
                //   n_accepted < k-1  → rollback (set n_past lower)
                //   n_accepted == k-1 → already correct
                //   n_accepted == k   → need one more decode (candidates[k-1])
                {
                    int draft_current = draft_P + (k - 1);
                    int draft_target  = draft_P + n_accepted;

                    if (draft_current > draft_target) {
                        // Rollback: discard rejected KV entries
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
                    // else: draft_current == draft_target, nothing to do

                    // Decode bonus on draft
                    if (!ar_forward_decode(draft_ctx, bonus,
                                           &draft_cache,
                                           prev_draft_logits.data())) {
                        DIFFUSE_LOG("speculative: draft decode bonus failed");
                        goto loop_done;
                    }
                }

                DIFFUSE_LOG("  spec: k=%d accepted=%d bonus=%d (%.0f%%)",
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
        DIFFUSE_LOG("  fast rejects: %d (d1 mismatch, no batch needed)",
                    stats.total_fast_rejects);
        DIFFUSE_LOG("  target calls: %d batches + %d decodes",
                    stats.total_target_batches, stats.total_target_decodes);
        DIFFUSE_LOG("  effective: %.2f tok/target_call",
                    stats.tokens_per_target_call());
        DIFFUSE_LOG("  total: %.1f ms", total_ms);
        DIFFUSE_LOG("  KV target: %d/%d pos (%.1f MB)",
                    target_cache.n_past, target_cache.n_ctx_max,
                    (float)target_cache.total_bytes() / (1024 * 1024));
        DIFFUSE_LOG("  KV draft:  %d/%d pos (%.1f MB)",
                    draft_cache.n_past, draft_cache.n_ctx_max,
                    (float)draft_cache.total_bytes() / (1024 * 1024));
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
