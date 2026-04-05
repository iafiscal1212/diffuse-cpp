#include "ar-graph.h"

// ── Helper: ensure tensor is F32 (for bias add after quantization) ──
static struct ggml_tensor * ensure_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Build autoregressive forward graph ──────────────────────────
//
// Standard transformer with causal attention and KV cache.
// Uses the same GGML ops as diffuse-graph.cpp but with:
//   1. Causal mask via ggml_soft_max_ext
//   2. KV cache append (only compute new tokens' K,V)
//   3. RoPE with absolute positions (n_past + i)
//
// n_new:  number of NEW tokens (full prompt for prefill, 1 for decode)
// n_past: number of tokens already in KV cache

static struct ggml_cgraph * ar_build_graph(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_new,
        int n_past,
        ar_kv_cache * cache) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int n_kv        = n_past + n_new;  // total KV length

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx,
            (size_t)(4096 * n_layer + n_layer * 4 + 256), false);

    // ── Input tokens ─────────────────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, tokens, n_new * sizeof(int32_t));

    // ── Position indices for new tokens ──────────────────────────
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    {
        int32_t * pos_data = (int32_t *)inp_pos->data;
        for (int i = 0; i < n_new; i++) {
            pos_data[i] = n_past + i;
        }
    }

    // ── Causal attention mask ────────────────────────────────────
    // Shape for ggml_soft_max_ext: [n_kv, n_new]
    // mask[k, q] = 0.0 if position k can be attended by query q, else -inf
    // For causal: q at absolute position (n_past+q) can see positions 0..(n_past+q)
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_new);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);
    {
        float * mask_data = (float *)attn_mask->data;
        for (int q = 0; q < n_new; q++) {
            int q_abs = n_past + q;
            for (int k = 0; k < n_kv; k++) {
                mask_data[q * n_kv + k] = (k <= q_abs) ? 0.0f : -INFINITY;
            }
        }
    }

    // ── Embedding lookup ─────────────────────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // ── Transformer layers ───────────────────────────────────────
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        // Pre-attention RMSNorm
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        // QKV projections for new tokens only
        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K_new = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V_new = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Qwen2.5)
        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K_new = ggml_add(ctx, K_new, ensure_f32(ctx, layer.bk));
        if (layer.bv) V_new = ggml_add(ctx, V_new, ensure_f32(ctx, layer.bv));

        // Reshape: [n_embd, n_new] → [n_embd_head, n_head(_kv), n_new]
        Q     = ggml_reshape_3d(ctx, Q,     n_embd_head, n_head,    n_new);
        K_new = ggml_reshape_3d(ctx, K_new, n_embd_head, n_head_kv, n_new);
        V_new = ggml_reshape_3d(ctx, V_new, n_embd_head, n_head_kv, n_new);

        // RoPE on Q and K_new with absolute positions
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_ext(ctx, K_new, inp_pos, nullptr, n_embd_head,
                              GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Mark K_new, V_new as outputs for cache extraction
        {
            char name_buf[32];
            snprintf(name_buf, sizeof(name_buf), "Kn.%02d", il);
            ggml_set_name(K_new, name_buf);
            ggml_set_output(K_new);

            snprintf(name_buf, sizeof(name_buf), "Vn.%02d", il);
            ggml_set_name(V_new, name_buf);
            ggml_set_output(V_new);
        }

        // ── Build full K,V: concat cached + new ─────────────────
        struct ggml_tensor * K_full;
        struct ggml_tensor * V_full;

        if (n_past > 0) {
            // Load cached K,V as input tensors
            // Shape: [n_embd_head, n_head_kv, n_past]
            struct ggml_tensor * K_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                    n_embd_head, n_head_kv, n_past);
            ggml_set_input(K_cached);
            memcpy(K_cached->data, cache->k_data(il),
                   (size_t)n_past * cache->pos_stride() * sizeof(float));

            struct ggml_tensor * V_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                    n_embd_head, n_head_kv, n_past);
            ggml_set_input(V_cached);
            memcpy(V_cached->data, cache->v_data(il),
                   (size_t)n_past * cache->pos_stride() * sizeof(float));

            K_full = ggml_concat(ctx, K_cached, K_new, 2);
            V_full = ggml_concat(ctx, V_cached, V_new, 2);
        } else {
            K_full = K_new;
            V_full = V_new;
        }
        // K_full, V_full: [n_embd_head, n_head_kv, n_kv]

        // ── GQA: expand KV heads to match Q heads ───────────────
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K_full = ggml_reshape_4d(ctx, K_full, n_embd_head, 1, n_head_kv, n_kv);
            K_full = ggml_repeat(ctx, K_full,
                    ggml_new_tensor_4d(ctx, K_full->type, n_embd_head, n_rep, n_head_kv, n_kv));
            K_full = ggml_reshape_3d(ctx, K_full, n_embd_head, n_head, n_kv);

            V_full = ggml_reshape_4d(ctx, V_full, n_embd_head, 1, n_head_kv, n_kv);
            V_full = ggml_repeat(ctx, V_full,
                    ggml_new_tensor_4d(ctx, V_full->type, n_embd_head, n_rep, n_head_kv, n_kv));
            V_full = ggml_reshape_3d(ctx, V_full, n_embd_head, n_head, n_kv);
        }

        // ── Attention (same pattern as diffuse-graph.cpp) ────────
        // Q: [n_embd_head, n_head, n_new] → [n_embd_head, n_new, n_head]
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        // K: [n_embd_head, n_head, n_kv] → [n_embd_head, n_kv, n_head]
        K_full = ggml_permute(ctx, K_full, 0, 2, 1, 3);
        // V: [n_embd_head, n_head, n_kv] → [n_kv, n_embd_head, n_head] (contiguous)
        V_full = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

        // K^T @ Q → [n_kv, n_new, n_head]
        struct ggml_tensor * attn = ggml_mul_mat(ctx, K_full, Q);

        // Scale + causal mask + softmax (fused)
        float scale = 1.0f / sqrtf((float)n_embd_head);
        attn = ggml_soft_max_ext(ctx, attn, attn_mask, scale, 0.0f);

        // V^T @ attn → [n_embd_head, n_new, n_head]
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, V_full, attn);

        // Merge heads: [n_embd_head, n_new, n_head] → [n_embd_head, n_head, n_new] → [n_embd, n_new]
        attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, n_new);

        // Output projection
        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN (SwiGLU) ─────────────────────────────────────────
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.ffn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur);
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, layer.ffn_down, cur);

        cur = ggml_add(ctx, cur, residual);
    }

    // ── Final norm + logits ──────────────────────────────────────
    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) {
            norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx, cur, norm_w);
    }
    cur = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ── Execute forward pass and extract K,V into cache ──────────────
static bool ar_forward_impl(
        diffuse_context * ctx,
        const int32_t * tokens,
        int n_new,
        ar_kv_cache * cache,
        float * logits_out) {

    const auto & hp = ctx->model->hparams;
    const int n_past = cache->n_past;
    const int n_kv   = n_past + n_new;
    const int n_head = (int)hp.n_head;

    // ── Compute buffer sizing ────────────────────────────────────
    size_t per_layer = (size_t)n_new * hp.n_embd * sizeof(float) * 10
                     + (size_t)n_new * hp.n_ff   * sizeof(float) * 3
                     // Attention matrix: [n_kv, n_new, n_head] after GQA expansion
                     + (size_t)n_kv * n_new * n_head * sizeof(float) * 2
                     // Cached K,V input: [n_embd_head, n_head_kv, n_past] × 2
                     + (size_t)n_past * hp.n_embd_head() * hp.n_head_kv * sizeof(float) * 2
                     // Expanded K,V: [n_embd_head, n_head, n_kv] × 2
                     + (size_t)n_kv * hp.n_embd_head() * n_head * sizeof(float) * 4
                     + hp.n_embd * sizeof(float) * 2;
    size_t buf_size = per_layer * hp.n_layer;

    // Logits buffer
    buf_size += (size_t)n_new * hp.n_vocab * sizeof(float) * 2;

    // Mask tensor + positions
    buf_size += (size_t)n_kv * n_new * sizeof(float);
    buf_size += n_new * sizeof(int32_t) * 2;

    // Overhead
    buf_size += 256ull * 1024 * 1024;
    buf_size = (size_t)(buf_size * 1.3);

    struct ggml_init_params cparams = {
        /*.mem_size   = */ buf_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    struct ggml_context * ctx_compute = ggml_init(cparams);
    if (!ctx_compute) {
        DIFFUSE_LOG("ar_forward: failed to allocate compute context (%zu MB)",
                    buf_size / (1024 * 1024));
        return false;
    }

    struct ggml_cgraph * gf = ar_build_graph(ctx, ctx_compute, tokens,
                                              n_new, n_past, cache);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        DIFFUSE_LOG("ar_forward: graph compute failed with status %d", (int)status);
        ggml_free(ctx_compute);
        return false;
    }

    // Extract logits
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("ar_forward: logits tensor not found");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_new * hp.n_vocab * sizeof(float));

    // Extract K_new, V_new and append to cache
    {
        char name_buf[32];
        for (int il = 0; il < (int)hp.n_layer; il++) {
            snprintf(name_buf, sizeof(name_buf), "Kn.%02d", il);
            struct ggml_tensor * K_t = ggml_graph_get_tensor(gf, name_buf);
            snprintf(name_buf, sizeof(name_buf), "Vn.%02d", il);
            struct ggml_tensor * V_t = ggml_graph_get_tensor(gf, name_buf);

            if (K_t && V_t) {
                cache->append(il, (const float *)K_t->data,
                                  (const float *)V_t->data, n_new);
            } else {
                DIFFUSE_LOG("WARNING: could not extract K/V for layer %d", il);
            }
        }
    }
    cache->advance(n_new);

    ggml_free(ctx_compute);
    return true;
}

// ── Public API: Prefill ──────────────────────────────────────────
bool ar_forward_prefill(
        diffuse_context * ctx,
        const int32_t * tokens,
        int n_tokens,
        ar_kv_cache * cache,
        float * logits_out) {

    if (!cache || !cache->initialized) {
        DIFFUSE_LOG("ar_forward_prefill: cache not initialized");
        return false;
    }

    cache->reset();
    DIFFUSE_LOG("ar_forward_prefill: %d tokens", n_tokens);
    return ar_forward_impl(ctx, tokens, n_tokens, cache, logits_out);
}

// ── Public API: Decode one token ─────────────────────────────────
bool ar_forward_decode(
        diffuse_context * ctx,
        int32_t token,
        ar_kv_cache * cache,
        float * logits_out) {

    if (!cache || !cache->initialized) {
        DIFFUSE_LOG("ar_forward_decode: cache not initialized");
        return false;
    }

    return ar_forward_impl(ctx, &token, 1, cache, logits_out);
}
