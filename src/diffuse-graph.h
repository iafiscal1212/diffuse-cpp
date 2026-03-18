#pragma once

#include "diffuse-common.h"

// Build the full transformer forward pass as a GGML computation graph.
// Returns the logits tensor [n_tokens, n_vocab].
struct ggml_cgraph * diffuse_build_graph(
    diffuse_context * ctx,
    struct ggml_context * ctx_compute,
    const int32_t * tokens,
    int n_tokens);
