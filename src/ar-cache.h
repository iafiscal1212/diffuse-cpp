#pragma once

#include "diffuse-common.h"
#include <vector>
#include <cstring>

// ── Autoregressive KV cache (append-only) ────────────────────────
//
// Standard autoregressive KV cache: appends new K,V after each token.
// Layout per layer: [n_ctx_max][n_head_kv * n_embd_head] flattened.
//
// For Qwen2.5-32B: 64 layers, 8 KV heads, 128 dim, 4096 ctx max
//   = 64 * 2 * 4096 * 8 * 128 * 4 bytes = ~2 GB
//
// K and V are stored BEFORE GQA expansion (n_head_kv, not n_head)
// to save memory. GQA repeat happens in the graph at attention time.

struct ar_kv_cache {
    bool initialized = false;
    int n_ctx_max   = 0;   // max context length
    int n_past      = 0;   // tokens currently in cache
    int n_layer     = 0;
    int n_embd_head = 0;
    int n_head_kv   = 0;   // KV heads (8 for Qwen2.5-32B, not expanded)

    // Per-layer K and V arrays
    // Shape per layer: [n_ctx_max * n_head_kv * n_embd_head]
    // Access pattern: cache[layer][pos * stride ... (pos+1) * stride]
    std::vector<std::vector<float>> K;  // K[layer]
    std::vector<std::vector<float>> V;  // V[layer]

    // ── Lifecycle ────────────────────────────────────────────────

    void init(int ctx_max, int n_lay, int n_eh, int n_hkv) {
        n_ctx_max   = ctx_max;
        n_layer     = n_lay;
        n_embd_head = n_eh;
        n_head_kv   = n_hkv;
        n_past      = 0;

        size_t kv_size = (size_t)n_ctx_max * n_head_kv * n_embd_head;
        K.resize(n_layer);
        V.resize(n_layer);
        for (int il = 0; il < n_layer; il++) {
            K[il].resize(kv_size, 0.0f);
            V[il].resize(kv_size, 0.0f);
        }

        initialized = true;
        DIFFUSE_LOG("ar_kv_cache: %d layers, %d kv_heads, %d head_dim, %d ctx_max (%.1f MB)",
                    n_layer, n_head_kv, n_embd_head, n_ctx_max,
                    (float)(2 * n_layer * kv_size * sizeof(float)) / (1024 * 1024));
    }

    void clear() {
        K.clear();
        V.clear();
        n_past = 0;
        initialized = false;
    }

    // Reset to empty (keep allocation)
    void reset() {
        n_past = 0;
    }

    // Floats per position per layer: n_head_kv * n_embd_head
    size_t pos_stride() const {
        return (size_t)n_head_kv * n_embd_head;
    }

    // Total bytes of the cache
    size_t total_bytes() const {
        return (size_t)2 * n_layer * n_ctx_max * pos_stride() * sizeof(float);
    }

    // ── Append new K,V for n_new tokens at positions [n_past, n_past+n_new) ──
    //
    // K_new, V_new: [n_embd_head, n_head_kv, n_new] (GGML layout, pre-GQA)
    // This matches the output shape from QKV projection + reshape_3d.

    void append(int layer, const float * K_new, const float * V_new, int n_new) {
        if (n_past + n_new > n_ctx_max) {
            DIFFUSE_LOG("WARNING: KV cache overflow (%d + %d > %d), wrapping",
                        n_past, n_new, n_ctx_max);
            // Simple sliding window: shift left by n_new
            size_t stride = pos_stride();
            size_t shift_bytes = (size_t)(n_ctx_max - n_new) * stride * sizeof(float);
            size_t new_bytes   = (size_t)n_new * stride * sizeof(float);
            memmove(K[layer].data(), K[layer].data() + n_new * stride, shift_bytes);
            memmove(V[layer].data(), V[layer].data() + n_new * stride, shift_bytes);
            memcpy(K[layer].data() + (n_ctx_max - n_new) * stride, K_new, new_bytes);
            memcpy(V[layer].data() + (n_ctx_max - n_new) * stride, V_new, new_bytes);
            // n_past stays at n_ctx_max (updated by caller only once after all layers)
            return;
        }

        size_t stride = pos_stride();
        size_t offset = (size_t)n_past * stride;
        size_t bytes  = (size_t)n_new * stride * sizeof(float);
        memcpy(K[layer].data() + offset, K_new, bytes);
        memcpy(V[layer].data() + offset, V_new, bytes);
    }

    // Advance the position counter (call once after appending to all layers)
    void advance(int n_new) {
        n_past = std::min(n_past + n_new, n_ctx_max);
    }

    // ── Access cached K,V for attention ──────────────────────────
    //
    // Returns pointer to layer's K or V data starting at position 0.
    // Caller uses n_past to know how many positions are valid.

    const float * k_data(int layer) const { return K[layer].data(); }
    const float * v_data(int layer) const { return V[layer].data(); }
};
