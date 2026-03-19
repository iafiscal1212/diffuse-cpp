#include "diffuse-sampler.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

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

// ── Iterative unmasking sampler ────────────────────────────────
std::vector<int32_t> diffuse_sample(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {

    const auto & hp = ctx->model->hparams;
    const int mask_id = hp.mask_token_id;
    const int n_vocab = hp.n_vocab;

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

    // Logits buffer
    std::vector<float> logits(total_len * n_vocab);

    // RNG for stochastic sampling
    std::mt19937 rng(params.seed);

    DIFFUSE_LOG("diffusion: %d steps, %d tokens to generate, scheduler=%s",
                params.n_steps, n_generate,
                params.remasking == diffuse_remasking::ENTROPY_EXIT ? "entropy_exit" :
                params.remasking == diffuse_remasking::LOW_CONFIDENCE ? "low_confidence" :
                "random");

    for (int step = 0; step < params.n_steps && n_masked > 0; step++) {
        // Forward pass
        if (!diffuse_forward(ctx, seq.data(), total_len, logits.data())) {
            DIFFUSE_DIE("forward pass failed at step %d", step);
        }

        // For each masked position, compute candidate token, confidence, and entropy
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

            const float * logit_row = logits.data() + (size_t)i * n_vocab;
            float ent = compute_entropy(logit_row, n_vocab);

            if (params.temperature <= 0.0f) {
                // Argmax
                int best = 0;
                float best_val = logit_row[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logit_row[v] > best_val) {
                        best_val = logit_row[v];
                        best = v;
                    }
                }
                candidates.push_back({i, best, best_val, ent});
            } else {
                // Temperature sampling
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

        // Determine how many tokens to unmask this step
        int n_unmask = tokens_to_unmask(step, params.n_steps, n_masked, params.schedule);

        if (params.remasking == diffuse_remasking::ENTROPY_EXIT) {
            // Entropy-exit: unmask all low-entropy tokens + scheduled amount
            // Sort by entropy ascending (easiest first)
            std::sort(candidates.begin(), candidates.end(),
                      [](const candidate & a, const candidate & b) {
                          return a.entropy < b.entropy;
                      });

            // Count tokens below entropy threshold
            int n_easy = 0;
            for (const auto & c : candidates) {
                if (c.entropy < params.entropy_threshold) n_easy++;
                else break;  // sorted, so no need to continue
            }

            // Unmask at least n_unmask, but also all easy tokens
            n_unmask = std::max(n_unmask, n_easy);
            n_unmask = std::min(n_unmask, (int)candidates.size());
        } else if (params.remasking == diffuse_remasking::LOW_CONFIDENCE) {
            // Sort by confidence descending → unmask the most confident
            std::sort(candidates.begin(), candidates.end(),
                      [](const candidate & a, const candidate & b) {
                          return a.confidence > b.confidence;
                      });
        } else {
            // Random order
            std::shuffle(candidates.begin(), candidates.end(), rng);
        }

        // Unmask top-n_unmask
        int actually_unmasked = 0;
        for (int j = 0; j < n_unmask && j < (int)candidates.size(); j++) {
            int pos = candidates[j].pos;
            seq[pos] = candidates[j].token;
            is_masked[pos] = false;
            n_masked--;
            actually_unmasked++;
        }

        DIFFUSE_LOG("  step %d/%d: unmasked %d tokens, %d remaining",
                    step + 1, params.n_steps, actually_unmasked, n_masked);

        if (callback) {
            callback(step + 1, params.n_steps, seq);
        }
    }

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
