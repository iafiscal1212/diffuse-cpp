#include "ar-sampler.h"
#include "ar-graph.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>

// ── Top-K filtering ─────────────────────────────────────────────
// Keep only the top_k highest logits, set rest to -inf.
static void apply_top_k(float * logits, int n_vocab, int top_k) {
    if (top_k <= 0 || top_k >= n_vocab) return;

    // Partial sort to find the k-th largest value
    std::vector<float> tmp(logits, logits + n_vocab);
    std::nth_element(tmp.begin(), tmp.begin() + top_k, tmp.end(),
                     std::greater<float>());
    float threshold = tmp[top_k - 1];

    // Zero out anything below threshold
    // (keep ties, so count how many are >= threshold)
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}

// ── Top-P (nucleus) filtering ───────────────────────────────────
// Sort by probability, keep tokens until cumulative prob >= top_p.
static void apply_top_p(float * logits, int n_vocab, float top_p) {
    if (top_p >= 1.0f) return;

    // Softmax to get probabilities
    float max_val = *std::max_element(logits, logits + n_vocab);
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < n_vocab; i++) {
        logits[i] /= sum;
    }

    // Sort indices by probability (descending)
    std::vector<int> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    // Accumulate until we hit top_p
    float cum = 0.0f;
    int cutoff = n_vocab;
    for (int i = 0; i < n_vocab; i++) {
        cum += logits[idx[i]];
        if (cum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything beyond cutoff
    for (int i = cutoff; i < n_vocab; i++) {
        logits[idx[i]] = 0.0f;
    }

    // Convert back to log-space for temperature application
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = (logits[i] > 0.0f) ? logf(logits[i]) : -INFINITY;
    }
}

// ── Repetition penalty ──────────────────────────────────────────
static void apply_repeat_penalty(float * logits, int n_vocab,
                                  const std::vector<int32_t> & recent_tokens,
                                  float penalty) {
    if (penalty <= 1.0f) return;

    for (int32_t tok : recent_tokens) {
        if (tok >= 0 && tok < n_vocab) {
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

// ── Sample from logits ──────────────────────────────────────────
static int32_t sample_token(float * logits, int n_vocab,
                             const ar_sampler_params & params,
                             const std::vector<int32_t> & recent_tokens,
                             std::mt19937 & rng) {

    // Apply repetition penalty
    apply_repeat_penalty(logits, n_vocab, recent_tokens, params.repeat_penalty);

    // Greedy (temperature == 0)
    if (params.temperature <= 0.0f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Apply temperature
    for (int i = 0; i < n_vocab; i++) {
        logits[i] /= params.temperature;
    }

    // Apply top-k
    apply_top_k(logits, n_vocab, params.top_k);

    // Apply top-p
    apply_top_p(logits, n_vocab, params.top_p);

    // Softmax → sample
    float max_val = *std::max_element(logits, logits + n_vocab);
    float sum = 0.0f;
    std::vector<float> probs(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; i++) {
        probs[i] /= sum;
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ── Main autoregressive generation loop ─────────────────────────
std::vector<int32_t> ar_generate(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int max_tokens,
        const ar_sampler_params & params,
        ar_token_callback callback) {

    const auto & hp = ctx->model->hparams;
    const int n_vocab = hp.n_vocab;
    const int prompt_len = (int)prompt_tokens.size();
    const int total_ctx = prompt_len + max_tokens;

    DIFFUSE_LOG("ar_generate: prompt=%d tokens, max_generate=%d, ctx=%d",
                prompt_len, max_tokens, total_ctx);

    // Initialize KV cache
    ar_kv_cache cache;
    cache.init(total_ctx, (int)hp.n_layer, (int)hp.n_embd_head(),
               (int)hp.n_head_kv);

    // RNG
    std::mt19937 rng(params.seed);

    // Logits buffer
    std::vector<float> logits;

    // Timing
    using clk = std::chrono::steady_clock;
    auto gen_start = clk::now();
    double prefill_ms = 0.0;
    double decode_ms  = 0.0;
    int n_decoded = 0;

    // ── Phase 1: Prefill (process all prompt tokens) ─────────────
    {
        auto t0 = clk::now();
        logits.resize((size_t)prompt_len * n_vocab);

        if (!ar_forward_prefill(ctx, prompt_tokens.data(), prompt_len,
                                &cache, logits.data())) {
            DIFFUSE_DIE("ar_generate: prefill failed");
        }

        auto t1 = clk::now();
        prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        DIFFUSE_LOG("  prefill: %d tokens in %.1f ms (%.1f tok/s)",
                    prompt_len, prefill_ms,
                    1000.0 * prompt_len / prefill_ms);
    }

    // ── Phase 2: Decode (token by token) ─────────────────────────
    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    // Recent tokens for repeat penalty (start with prompt tail)
    std::vector<int32_t> recent;
    {
        int start = std::max(0, prompt_len - params.repeat_last_n);
        for (int i = start; i < prompt_len; i++) {
            recent.push_back(prompt_tokens[i]);
        }
    }

    // Get the first token from prefill logits (last position)
    {
        float * last_logits = logits.data() + (size_t)(prompt_len - 1) * n_vocab;
        std::vector<float> logit_copy(last_logits, last_logits + n_vocab);
        int32_t first_token = sample_token(logit_copy.data(), n_vocab,
                                            params, recent, rng);
        generated.push_back(first_token);
        recent.push_back(first_token);
        if ((int)recent.size() > params.repeat_last_n) {
            recent.erase(recent.begin());
        }

        if (callback && !callback(first_token, prompt_len)) {
            goto done;
        }
    }

    // Decode remaining tokens
    logits.resize(n_vocab);  // Reuse for single-token decode

    {
        auto decode_start = clk::now();

        for (int i = 1; i < max_tokens; i++) {
            int32_t last_token = generated.back();

            // Check for EOS
            // Common EOS tokens: Qwen2.5 uses 151643 (<|im_end|>), 151645 (<|endoftext|>)
            // We check via the model's hparams if available, otherwise use hardcoded
            // For now, stop on any of the common Qwen2.5 stop tokens
            if (last_token == 151643 || last_token == 151645 ||
                last_token == 151644) {  // <|im_start|> shouldn't appear in generation
                DIFFUSE_LOG("  EOS token %d at position %d", last_token,
                            prompt_len + i - 1);
                break;
            }

            if (!ar_forward_decode(ctx, last_token, &cache, logits.data())) {
                DIFFUSE_LOG("ar_generate: decode failed at token %d", i);
                break;
            }

            std::vector<float> logit_copy(logits.begin(), logits.end());
            int32_t new_token = sample_token(logit_copy.data(), n_vocab,
                                              params, recent, rng);
            generated.push_back(new_token);
            recent.push_back(new_token);
            if ((int)recent.size() > params.repeat_last_n) {
                recent.erase(recent.begin());
            }
            n_decoded++;

            if (callback && !callback(new_token, prompt_len + i)) {
                break;
            }
        }

        auto decode_end = clk::now();
        decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    }

done:
    // ── Timing summary ───────────────────────────────────────────
    auto gen_end = clk::now();
    double total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();

    DIFFUSE_LOG("=== AR GENERATION COMPLETE ===");
    DIFFUSE_LOG("  prefill:  %6.1f ms  (%d tokens, %.1f tok/s)",
                prefill_ms, prompt_len,
                1000.0 * prompt_len / std::max(prefill_ms, 0.1));
    DIFFUSE_LOG("  decode:   %6.1f ms  (%d tokens, %.2f tok/s)",
                decode_ms, n_decoded,
                n_decoded > 0 ? 1000.0 * n_decoded / decode_ms : 0.0);
    DIFFUSE_LOG("  total:    %6.1f ms  (%d generated)",
                total_ms, (int)generated.size());
    DIFFUSE_LOG("  KV cache: %d / %d positions used (%.1f MB)",
                cache.n_past, cache.n_ctx_max,
                (float)cache.total_bytes() / (1024 * 1024));

    cache.clear();
    return generated;
}
