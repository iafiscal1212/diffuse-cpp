#pragma once

#include "diffuse-common.h"
#include <vector>
#include <cstring>

// ── Inter-step KV cache for diffusion denoising ─────────────────
//
// Caches K and V tensors (after projection + RoPE + GQA expansion)
// for positions that don't need recomputation between steps.
//
// Layout per layer: [n_embd_head, n_head, n_tokens] flattened to float array.
// This matches the GGML tensor layout BEFORE the permute step in attention.

struct diffuse_step_cache {
    bool initialized = false;
    int n_tokens    = 0;  // total sequence length (prompt + gen)
    int n_prompt    = 0;  // prompt token count
    int n_layer     = 0;
    int n_embd_head = 0;
    int n_head      = 0;

    // Per-layer cached K and V after RoPE + GQA expansion
    // Shape: [n_embd_head * n_head * n_tokens] per layer
    std::vector<std::vector<float>> K;  // K[layer]
    std::vector<std::vector<float>> V;  // V[layer]

    // Token sequence from previous step (to detect changes)
    std::vector<int32_t> prev_seq;

    // Step at which each position last changed token (for extended active set)
    std::vector<int> last_changed_step;

    // ── Lifecycle ────────────────────────────────────────────────

    void init(int n_tok, int n_prom, int n_lay, int n_eh, int n_h) {
        n_tokens    = n_tok;
        n_prompt    = n_prom;
        n_layer     = n_lay;
        n_embd_head = n_eh;
        n_head      = n_h;

        size_t kv_size = (size_t)n_embd_head * n_head * n_tokens;
        K.resize(n_layer);
        V.resize(n_layer);
        for (int il = 0; il < n_layer; il++) {
            K[il].resize(kv_size, 0.0f);
            V[il].resize(kv_size, 0.0f);
        }

        prev_seq.resize(n_tokens, -1);
        last_changed_step.resize(n_tokens, -1);
        initialized = false;
    }

    void clear() {
        K.clear();
        V.clear();
        prev_seq.clear();
        last_changed_step.clear();
        initialized = false;
    }

    // Bytes per position in K or V: n_embd_head * n_head * sizeof(float)
    size_t pos_stride() const {
        return (size_t)n_embd_head * n_head;
    }

    // ── Active set computation ───────────────────────────────────
    //
    // Active positions = positions whose logits we need OR whose
    // embeddings changed since the last step OR that changed recently
    // (within extra_active_steps).
    //
    // Specifically: {still masked} ∪ {changed since last step}
    //             ∪ {changed within last extra_active_steps steps}
    // Cached positions: all others (prompt + stably unmasked)

    void compute_active_set(
            const int32_t * seq,
            const std::vector<bool> & is_masked,
            int n_total,
            int current_step,
            int extra_active_steps,
            // outputs:
            std::vector<int> & cached_positions,   // original indices, sorted
            std::vector<int> & active_positions,    // original indices, sorted
            std::vector<int> & active_to_orig) {    // active_idx → original position

        cached_positions.clear();
        active_positions.clear();

        for (int i = 0; i < n_total; i++) {
            bool token_changed = (seq[i] != prev_seq[i]);
            bool needs_logits  = is_masked[i];

            // Track when this position last changed
            if (token_changed) {
                last_changed_step[i] = current_step;
            }

            // Recently changed: keep active for extra steps to refresh stale K,V
            bool recently_changed = (extra_active_steps > 0 &&
                                     last_changed_step[i] >= 0 &&
                                     (current_step - last_changed_step[i]) <= extra_active_steps);

            if (token_changed || needs_logits || recently_changed) {
                active_positions.push_back(i);
            } else {
                cached_positions.push_back(i);
            }
        }

        // active_to_orig is just active_positions (they're already original indices)
        active_to_orig = active_positions;
    }

    // ── Cache update ─────────────────────────────────────────────
    //
    // After a forward pass, store K and V for active positions into the cache.
    // K_active_data: [n_embd_head, n_head, n_active] from the graph output
    // active_positions: original position indices for each active token

    void update_kv(int layer,
                   const float * K_active_data,
                   const float * V_active_data,
                   const std::vector<int> & active_positions) {

        size_t stride = pos_stride();  // floats per position

        for (int a = 0; a < (int)active_positions.size(); a++) {
            int orig_pos = active_positions[a];
            memcpy(K[layer].data() + orig_pos * stride,
                   K_active_data + a * stride,
                   stride * sizeof(float));
            memcpy(V[layer].data() + orig_pos * stride,
                   V_active_data + a * stride,
                   stride * sizeof(float));
        }
    }

    // Store the current sequence for next step's change detection
    void update_seq(const int32_t * seq, int n) {
        memcpy(prev_seq.data(), seq, n * sizeof(int32_t));
        initialized = true;
    }
};
