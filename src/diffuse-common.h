#pragma once

#include "diffuse.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

// ── Logging ────────────────────────────────────────────────────
#define DIFFUSE_LOG(fmt, ...) fprintf(stderr, "[diffuse] " fmt "\n", ##__VA_ARGS__)
#define DIFFUSE_DIE(fmt, ...) do { \
    fprintf(stderr, "[diffuse] FATAL: " fmt "\n", ##__VA_ARGS__); \
    exit(1); \
} while(0)

// ── Tensor name helpers ────────────────────────────────────────
static inline std::string fmt_layer(const char * pattern, int i) {
    char buf[128];
    snprintf(buf, sizeof(buf), pattern, i);
    return buf;
}

// ── Per-layer weight struct ────────────────────────────────────
struct diffuse_layer {
    // Attention
    struct ggml_tensor * attn_norm;   // RMSNorm weight
    struct ggml_tensor * wq;          // Q projection
    struct ggml_tensor * wk;          // K projection
    struct ggml_tensor * wv;          // V projection
    struct ggml_tensor * wo;          // output projection

    // FFN (SwiGLU)
    struct ggml_tensor * ffn_norm;    // RMSNorm weight
    struct ggml_tensor * ffn_gate;    // gate projection (w1)
    struct ggml_tensor * ffn_up;      // up projection (w3)
    struct ggml_tensor * ffn_down;    // down projection (w2)
};

// ── Full model struct ──────────────────────────────────────────
struct diffuse_model {
    diffuse_hparams hparams;

    // Embeddings
    struct ggml_tensor * tok_embd;    // token embeddings
    struct ggml_tensor * output_norm; // final RMSNorm
    struct ggml_tensor * output;      // lm_head (may be tied to tok_embd)

    // Layers
    std::vector<diffuse_layer> layers;

    // GGML backend
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf     = nullptr;
    struct ggml_context    * ctx     = nullptr;     // weight context
};

// ── Compute context ────────────────────────────────────────────
struct diffuse_context {
    const diffuse_model * model;
    int n_ctx;
    int n_threads;

    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    struct ggml_context  * ctx    = nullptr;     // compute context
};
