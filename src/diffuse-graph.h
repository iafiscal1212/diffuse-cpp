#pragma once

#include "diffuse-common.h"
#include "diffuse-cache.h"
#include <vector>

// Build the full transformer forward pass as a GGML computation graph.
// Returns the logits tensor [n_tokens, n_vocab].
struct ggml_cgraph * diffuse_build_graph(
    diffuse_context * ctx,
    struct ggml_context * ctx_compute,
    const int32_t * tokens,
    int n_tokens);

// Full forward pass. If cache is non-null, extracts K/V into cache.
// logits_out: [n_tokens * n_vocab]
bool diffuse_forward_full(
    diffuse_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    float * logits_out,
    diffuse_step_cache * cache);

// Original forward pass (no cache). Backward compatible.
bool diffuse_forward(
    diffuse_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    float * logits_out);

// Cached forward: computes only for active positions.
// logits_out: [n_active * n_vocab]
bool diffuse_forward_cached(
    diffuse_context * ctx,
    const int32_t * active_tokens,
    const int32_t * active_pos_indices,
    int n_active,
    int n_total,
    diffuse_step_cache * cache,
    const std::vector<int> & cached_positions,
    const std::vector<int> & active_positions,
    float * logits_out);
