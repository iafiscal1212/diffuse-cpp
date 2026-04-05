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
//
// Also holds persistent compute resources (buffer, threadpool, work data)
// to avoid malloc/free and thread creation overhead each decode step.

struct ggml_threadpool;  // forward declaration

struct ar_kv_cache {
    bool initialized = false;
    int n_ctx_max   = 0;   // max context length
    int n_past      = 0;   // tokens currently in cache
    int n_layer     = 0;
    int n_embd_head = 0;
    int n_head_kv   = 0;   // KV heads (8 for Qwen2.5-32B, not expanded)

    // ── Layer skip / optimization (VSIDS-inspired) ──────────────
    std::vector<bool> skip_layers;       // [n_layer] true = skip during decode
    std::vector<float> layer_impact;     // [n_layer] profiled impact scores
    int sliding_window = 0;              // 0 = full context, >0 = attend last W positions

    // Per-layer K and V arrays
    // Shape per layer: [n_ctx_max * n_head_kv * n_embd_head]
    // Access pattern: cache[layer][pos * stride ... (pos+1) * stride]
    std::vector<std::vector<float>> K;  // K[layer]
    std::vector<std::vector<float>> V;  // V[layer]

    // ── Persistent compute resources (avoid malloc/free per step) ──
    void *   compute_buf      = nullptr;   // pre-allocated GGML context buffer
    size_t   compute_buf_size = 0;
    uint8_t * work_buf        = nullptr;   // pre-allocated graph work buffer
    size_t   work_buf_size    = 0;
    struct ggml_threadpool * threadpool = nullptr;  // persistent thread pool

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
        skip_layers.clear();
        layer_impact.clear();
        n_past = 0;
        sliding_window = 0;
        initialized = false;
        free_compute_resources();
    }

    void free_compute_resources() {
        if (compute_buf) { free(compute_buf); compute_buf = nullptr; compute_buf_size = 0; }
        if (work_buf)    { free(work_buf);    work_buf = nullptr;    work_buf_size = 0;    }
        // threadpool freed externally (owned by caller who created it)
        threadpool = nullptr;
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

    // ── Ensure compute buffer is large enough ─────────────────────
    void ensure_compute_buf(size_t needed) {
        if (needed <= compute_buf_size) return;
        free(compute_buf);
        compute_buf_size = needed;
        compute_buf = malloc(compute_buf_size);
        if (!compute_buf) {
            DIFFUSE_DIE("ar_kv_cache: failed to allocate compute buffer (%zu MB)",
                        compute_buf_size / (1024 * 1024));
        }
    }

    // ── Ensure work buffer is large enough ────────────────────────
    void ensure_work_buf(size_t needed) {
        if (needed <= work_buf_size) return;
        free(work_buf);
        work_buf_size = needed;
        work_buf = (uint8_t *)malloc(work_buf_size);
        if (!work_buf) {
            DIFFUSE_DIE("ar_kv_cache: failed to allocate work buffer (%zu MB)",
                        work_buf_size / (1024 * 1024));
        }
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
