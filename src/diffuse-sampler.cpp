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
        // Fraction of remaining masked tokens to unmask this step
    } else {
        // Linear: equal fraction each step
        frac = 1.0f / (total_steps - step);
    }

    int n = (int)roundf(frac * total_masked);
    if (n < 1 && total_masked > 0) n = 1;
    return std::min(n, total_masked);
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

    DIFFUSE_LOG("diffusion: %d steps, %d tokens to generate", params.n_steps, n_generate);

    for (int step = 0; step < params.n_steps && n_masked > 0; step++) {
        // Forward pass
        if (!diffuse_forward(ctx, seq.data(), total_len, logits.data())) {
            DIFFUSE_DIE("forward pass failed at step %d", step);
        }

        // For each masked position, compute candidate token and confidence
        struct candidate {
            int pos;
            int token;
            float confidence;
        };
        std::vector<candidate> candidates;
        candidates.reserve(n_masked);

        for (int i = 0; i < total_len; i++) {
            if (!is_masked[i]) continue;

            const float * logit_row = logits.data() + (size_t)i * n_vocab;

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
                // Confidence = softmax probability of best token (approximate)
                // Use max logit - second max as a proxy for speed
                candidates.push_back({i, best, best_val});
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

                // Sample from distribution
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int sampled = dist(rng);
                candidates.push_back({i, sampled, probs[sampled]});
            }
        }

        // Determine how many tokens to unmask this step
        int n_unmask = tokens_to_unmask(step, params.n_steps, n_masked, params.schedule);

        if (params.remasking == diffuse_remasking::LOW_CONFIDENCE) {
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
        for (int j = 0; j < n_unmask && j < (int)candidates.size(); j++) {
            int pos = candidates[j].pos;
            seq[pos] = candidates[j].token;
            is_masked[pos] = false;
            n_masked--;
        }

        DIFFUSE_LOG("  step %d/%d: unmasked %d tokens, %d remaining",
                    step + 1, params.n_steps, n_unmask, n_masked);

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
