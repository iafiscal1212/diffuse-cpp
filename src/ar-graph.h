#pragma once

#include "diffuse-common.h"
#include "ar-cache.h"

// ── Autoregressive forward pass ─────────────────────────────────
//
// Two modes:
//   1. Prefill: process N prompt tokens in batch, populate KV cache
//   2. Decode:  process 1 token, read from KV cache
//
// Both modes return logits only for the NEW tokens (not the full ctx).

// Prefill: process prompt tokens, fill KV cache, return logits for all positions.
// tokens: [n_tokens], logits_out: [n_tokens * n_vocab]
bool ar_forward_prefill(
    diffuse_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    ar_kv_cache * cache,
    float * logits_out);

// Decode: process a single new token, append to KV cache, return logits.
// token: single token ID, logits_out: [n_vocab]
bool ar_forward_decode(
    diffuse_context * ctx,
    int32_t token,
    ar_kv_cache * cache,
    float * logits_out);

// Batch: process N tokens with existing KV cache (no reset).
// Used for speculative decoding verification: target verifies K draft tokens at once.
// tokens: [n_tokens], logits_out: [n_tokens * n_vocab]
bool ar_forward_batch(
    diffuse_context * ctx,
    const int32_t * tokens,
    int n_tokens,
    ar_kv_cache * cache,
    float * logits_out);
