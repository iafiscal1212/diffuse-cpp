#include "diffuse-sampler.h"
#include "diffuse-graph.h"
#include "diffuse-cache.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>

// ── Masking schedule ───────────────────────────────────────────
static int tokens_to_unmask(int step, int total_steps, int total_masked,
                            diffuse_schedule schedule) {
    float t0 = (float)step / total_steps;
    float t1 = (float)(step + 1) / total_steps;

    float frac;
    if (schedule == diffuse_schedule::COSINE) {
        // Cosine schedule: more tokens unmasked in early steps
        float cos0 = cosf(t0 * M_PI * 0.5f);
        float cos1 = cosf(t1 * M_PI * 0.5f);
        frac = (cos0 - cos1) / cos0;
    } else {
        // Linear: equal fraction each step
        frac = 1.0f / (total_steps - step);
    }

    int n = (int)roundf(frac * total_masked);
    if (n < 1 && total_masked > 0) n = 1;
    return std::min(n, total_masked);
}

// ── Compute entropy of logit distribution ─────────────────────
static float compute_entropy(const float * logits, int n_vocab) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > max_val) max_val = logits[v];
    }

    // Softmax + entropy in one pass
    float sum_exp = 0.0f;
    for (int v = 0; v < n_vocab; v++) {
        sum_exp += expf(logits[v] - max_val);
    }
    float log_sum = logf(sum_exp);

    float entropy = 0.0f;
    for (int v = 0; v < n_vocab; v++) {
        float log_p = (logits[v] - max_val) - log_sum;
        float p = expf(log_p);
        if (p > 1e-10f) {
            entropy -= p * log_p;
        }
    }
    return entropy;
}

// ── Iterative unmasking sampler with inter-step caching ─────────
std::vector<int32_t> diffuse_sample(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {

    const auto & hp = ctx->model->hparams;
    const int mask_id = hp.mask_token_id;
    const int n_vocab = hp.n_vocab;

    // Dream (Qwen2.5 backbone) uses shifted logits: output at position i
    // predicts token at position i+1 (autoregressive convention).
    // To get logits for position i, read from position max(i-1, 0).
    const bool shift_logits = (ctx->model->model_type == "dream");

    // Build initial sequence: prompt + MASK tokens
    std::vector<int32_t> seq = prompt_tokens;
    int prompt_len = (int)prompt_tokens.size();
    seq.resize(prompt_len + n_generate, mask_id);
    int total_len = (int)seq.size();

    // Track which positions are still masked
    std::vector<bool> is_masked(total_len, false);
    for (int i = prompt_len; i < total_len; i++) {
        is_masked[i] = true;
    }
    int n_masked = n_generate;

    // Full logits buffer (only used for step 0)
    std::vector<float> logits_full(total_len * n_vocab);

    // RNG for stochastic sampling
    std::mt19937 rng(params.seed);

    // ── Inter-step KV cache ──────────────────────────────────────
    const bool use_cache = params.use_cache;
    diffuse_step_cache cache;
    if (use_cache) {
        cache.init(total_len, prompt_len,
                   (int)hp.n_layer, (int)hp.n_embd_head(), (int)hp.n_head);
    }

    const int cache_refresh = params.cache_refresh;
    const int cache_keep_active = params.cache_keep_active;

    const char * remasking_name =
        params.remasking == diffuse_remasking::ENTROPY_EXIT  ? "entropy_exit" :
        params.remasking == diffuse_remasking::LOW_CONFIDENCE ? "low_confidence" :
        params.remasking == diffuse_remasking::MASKGIT_PLUS  ? "maskgit_plus" :
        params.remasking == diffuse_remasking::TOPK_MARGIN   ? "topk_margin" :
        "random";
    DIFFUSE_LOG("diffusion: %d steps, %d tokens to generate, scheduler=%s, cache=%s, refresh=%d, keep_active=%d",
                params.n_steps, n_generate,
                remasking_name,
                use_cache ? "ON" : "OFF",
                cache_refresh, cache_keep_active);

    // ── Timing accumulators ──────────────────────────────────────
    using clk = std::chrono::steady_clock;
    double total_forward_ms  = 0.0;
    double total_sample_ms   = 0.0;
    double total_sort_ms     = 0.0;
    double total_unmask_ms   = 0.0;
    int    actual_steps      = 0;
    auto   gen_start         = clk::now();

    for (int step = 0; step < params.n_steps && n_masked > 0; step++) {
        actual_steps++;

        // ── Forward pass (timed) ─────────────────────────────────
        auto t0 = clk::now();

        // Logits pointer: where to read logits from after forward pass
        // For full forward: logits are in logits_full[pos * n_vocab]
        // For cached forward: logits are in logits_active[active_idx * n_vocab]
        float * logit_source = nullptr;

        // Active set tracking for cached forward
        std::vector<int> cached_positions, active_positions, active_to_orig;
        int n_active = 0;
        bool used_cache = false;

        // Force full forward on step 0, when cache disabled, or at refresh intervals
        bool force_full = !use_cache || !cache.initialized ||
                          (cache_refresh > 0 && step > 0 && (step % cache_refresh == 0));

        if (force_full) {
            // ── Full forward (step 0, refresh, or cache disabled) ─
            if (!diffuse_forward_full(ctx, seq.data(), total_len,
                                      logits_full.data(),
                                      use_cache ? &cache : nullptr)) {
                DIFFUSE_DIE("forward pass failed at step %d", step);
            }
            if (use_cache) cache.update_seq(seq.data(), total_len);
            logit_source = logits_full.data();
            n_active = total_len;  // all positions
        } else {
            // ── Steps 1+: cached forward, only active positions ──
            cache.compute_active_set(seq.data(), is_masked, total_len,
                                     step, cache_keep_active,
                                     cached_positions, active_positions,
                                     active_to_orig);
            n_active = (int)active_positions.size();

            if (n_active >= total_len || n_active == 0) {
                // Edge case: all active or none → full forward
                if (!diffuse_forward_full(ctx, seq.data(), total_len,
                                          logits_full.data(), &cache)) {
                    DIFFUSE_DIE("forward pass failed at step %d", step);
                }
                cache.update_seq(seq.data(), total_len);
                logit_source = logits_full.data();
                n_active = total_len;
            } else {
                // Build active token arrays
                std::vector<int32_t> active_tokens(n_active);
                std::vector<int32_t> active_pos_idx(n_active);
                for (int a = 0; a < n_active; a++) {
                    int orig = active_positions[a];
                    active_tokens[a] = seq[orig];
                    active_pos_idx[a] = orig;
                }

                // Allocate logits for active positions only
                std::vector<float> logits_active(n_active * n_vocab);

                if (!diffuse_forward_cached(
                        ctx, active_tokens.data(), active_pos_idx.data(),
                        n_active, total_len, &cache,
                        cached_positions, active_positions,
                        logits_active.data())) {
                    DIFFUSE_DIE("cached forward failed at step %d", step);
                }
                cache.update_seq(seq.data(), total_len);
                used_cache = true;

                // Scatter active logits into full logits buffer
                // We only need logits for masked positions, which are
                // always in the active set. Build a map: orig_pos → active_idx
                // for efficient lookup.
                // Actually, we can iterate only over masked positions
                // and find their index in active_positions.

                // For simplicity: store in logits_full at original positions
                for (int a = 0; a < n_active; a++) {
                    int orig = active_positions[a];
                    memcpy(logits_full.data() + (size_t)orig * n_vocab,
                           logits_active.data() + (size_t)a * n_vocab,
                           n_vocab * sizeof(float));
                }
                logit_source = logits_full.data();
            }
        }

        auto t1 = clk::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_forward_ms += fwd_ms;

        // ── Sampling: extract candidates (timed) ─────────────────
        auto t2 = clk::now();

        struct candidate {
            int pos;
            int token;
            float confidence;
            float entropy;
        };
        std::vector<candidate> candidates;
        candidates.reserve(n_masked);

        for (int i = 0; i < total_len; i++) {
            if (!is_masked[i]) continue;

            // Dream: shifted logits (position i uses logits from position max(i-1, 0))
            int logit_pos = shift_logits ? std::max(i - 1, 0) : i;
            const float * logit_row = logit_source + (size_t)logit_pos * n_vocab;
            float ent = compute_entropy(logit_row, n_vocab);

            if (params.temperature <= 0.0f) {
                // Find top-1 (and top-2 for TOPK_MARGIN)
                int best = 0;
                float best_val = logit_row[0];
                float second_val = -1e30f;
                for (int v = 1; v < n_vocab; v++) {
                    if (logit_row[v] > best_val) {
                        second_val = best_val;
                        best_val = logit_row[v];
                        best = v;
                    } else if (logit_row[v] > second_val) {
                        second_val = logit_row[v];
                    }
                }
                float conf = (params.remasking == diffuse_remasking::TOPK_MARGIN)
                           ? (best_val - second_val) : best_val;
                candidates.push_back({i, best, conf, ent});
            } else {
                float max_logit = *std::max_element(logit_row, logit_row + n_vocab);
                std::vector<float> probs(n_vocab);
                float sum = 0.0f;
                for (int v = 0; v < n_vocab; v++) {
                    probs[v] = expf((logit_row[v] - max_logit) / params.temperature);
                    sum += probs[v];
                }
                for (int v = 0; v < n_vocab; v++) probs[v] /= sum;

                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int sampled = dist(rng);
                candidates.push_back({i, sampled, probs[sampled], ent});
            }
        }
        auto t3 = clk::now();
        total_sample_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // ── Scheduling + sorting (timed) ─────────────────────────
        auto t4 = clk::now();

        int n_unmask = tokens_to_unmask(step, params.n_steps, n_masked, params.schedule);

        if (params.remasking == diffuse_remasking::ENTROPY_EXIT) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const candidate & a, const candidate & b) {
                          return a.entropy < b.entropy;
                      });

            int n_easy = 0;
            for (const auto & c : candidates) {
                if (c.entropy < params.entropy_threshold) n_easy++;
                else break;
            }

            n_unmask = std::max(n_unmask, n_easy);
            n_unmask = std::min(n_unmask, (int)candidates.size());
        } else if (params.remasking == diffuse_remasking::LOW_CONFIDENCE ||
                   params.remasking == diffuse_remasking::MASKGIT_PLUS) {
            // Both sort by highest confidence (top-1 logit value)
            std::sort(candidates.begin(), candidates.end(),
                      [](const candidate & a, const candidate & b) {
                          return a.confidence > b.confidence;
                      });
        } else if (params.remasking == diffuse_remasking::TOPK_MARGIN) {
            // Sort by margin between top-1 and top-2 logits (highest margin first)
            std::sort(candidates.begin(), candidates.end(),
                      [](const candidate & a, const candidate & b) {
                          return a.confidence > b.confidence;  // margin stored in confidence field
                      });
        } else {
            std::shuffle(candidates.begin(), candidates.end(), rng);
        }

        auto t5 = clk::now();
        total_sort_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        // ── Unmasking (timed) ────────────────────────────────────
        auto t6 = clk::now();

        int actually_unmasked = 0;
        for (int j = 0; j < n_unmask && j < (int)candidates.size(); j++) {
            int pos = candidates[j].pos;
            seq[pos] = candidates[j].token;
            is_masked[pos] = false;
            n_masked--;
            actually_unmasked++;
        }

        auto t7 = clk::now();
        total_unmask_ms += std::chrono::duration<double, std::milli>(t7 - t6).count();

        DIFFUSE_LOG("  step %d/%d: unmasked %d tokens, %d remaining "
                    "(fwd=%.1fms, active=%d/%d%s)",
                    step + 1, params.n_steps, actually_unmasked, n_masked,
                    fwd_ms,
                    used_cache ? (int)active_positions.size() : total_len,
                    total_len,
                    used_cache ? " CACHED" :
                    (force_full && step > 0) ? " REFRESH" : "");

        if (callback) {
            callback(step + 1, params.n_steps, seq);
        }
    }

    // ── Timing summary ───────────────────────────────────────────
    auto gen_end = clk::now();
    double gen_total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
    double overhead_ms  = gen_total_ms - total_forward_ms - total_sample_ms
                        - total_sort_ms - total_unmask_ms;

    DIFFUSE_LOG("=== TIMING BREAKDOWN (%d steps, cache=%s) ===",
                actual_steps, use_cache ? "ON" : "OFF");
    DIFFUSE_LOG("  forward:   %8.1f ms  (%5.1f%%)  avg=%.1f ms/step",
                total_forward_ms, 100.0 * total_forward_ms / gen_total_ms,
                total_forward_ms / std::max(actual_steps, 1));
    DIFFUSE_LOG("  sampling:  %8.1f ms  (%5.1f%%)  avg=%.1f ms/step",
                total_sample_ms, 100.0 * total_sample_ms / gen_total_ms,
                total_sample_ms / std::max(actual_steps, 1));
    DIFFUSE_LOG("  sorting:   %8.1f ms  (%5.1f%%)",
                total_sort_ms, 100.0 * total_sort_ms / gen_total_ms);
    DIFFUSE_LOG("  unmask:    %8.1f ms  (%5.1f%%)",
                total_unmask_ms, 100.0 * total_unmask_ms / gen_total_ms);
    DIFFUSE_LOG("  overhead:  %8.1f ms  (%5.1f%%)",
                overhead_ms, 100.0 * overhead_ms / gen_total_ms);
    DIFFUSE_LOG("  TOTAL:     %8.1f ms  (%.2f tok/s)",
                gen_total_ms, 1000.0 * n_generate / gen_total_ms);

    // ── Cleanup ──────────────────────────────────────────────────
    cache.clear();

    // Return only the generated tokens
    return std::vector<int32_t>(seq.begin() + prompt_len, seq.end());
}

// ── Public API wrapper ─────────────────────────────────────────
std::vector<int32_t> diffuse_generate(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {
    return diffuse_sample(ctx, prompt_tokens, n_generate, params, callback);
}
