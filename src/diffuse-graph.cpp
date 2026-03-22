#include "diffuse-graph.h"

// ── Helper: ensure tensor is F32 (for bias add after quantization) ──
static struct ggml_tensor * ensure_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Build transformer forward graph ────────────────────────────
struct ggml_cgraph * diffuse_build_graph(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_tokens) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int N           = n_tokens;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t)(4096 * n_layer), false);

    // ── Inputs ─────────────────────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, tokens, N * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    {
        int32_t * pos_data = (int32_t *)inp_pos->data;
        for (int i = 0; i < N; i++) pos_data[i] = i;
    }

    // ── Embedding lookup ───────────────────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // ── Transformer layers ─────────────────────────────────────
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

        // QKV projections
        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K = ggml_add(ctx, K, ensure_f32(ctx, layer.bk));
        if (layer.bv) V = ggml_add(ctx, V, ensure_f32(ctx, layer.bv));

        // Reshape for multi-head: [n_embd_head, n_head, N]
        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        // RoPE on Q and K (NEOX style = non-interleaved, like LLaDA/OLMo)
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat each KV head n_rep times (grouped, not interleaved)
        // Reshape 3D→4D so ggml_repeat expands the right axis, then flatten back.
        // K[d, n_kv, N] → K[d, 1, n_kv, N] → repeat → K[d, n_rep, n_kv, N] → K[d, n_head, N]
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K = ggml_reshape_4d(ctx, K, n_embd_head, 1, n_head_kv, N);
            K = ggml_repeat(ctx, K,
                    ggml_new_tensor_4d(ctx, K->type, n_embd_head, n_rep, n_head_kv, N));
            K = ggml_reshape_3d(ctx, K, n_embd_head, n_head, N);

            V = ggml_reshape_4d(ctx, V, n_embd_head, 1, n_head_kv, N);
            V = ggml_repeat(ctx, V,
                    ggml_new_tensor_4d(ctx, V->type, n_embd_head, n_rep, n_head_kv, N));
            V = ggml_reshape_3d(ctx, V, n_embd_head, n_head, N);
        }

        // Permute Q, K: [n_embd_head, n_head, N] → [n_embd_head, N, n_head]
        // ggml_permute semantics: result->ne[axis_i] = input->ne[i]
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);

        // V: [n_embd_head, n_head, N] → [N, n_embd_head, n_head] (contiguous)
        // Input dim 0→result dim 1, dim 1→result dim 2, dim 2→result dim 0
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

        // Attention: K^T @ Q → [N, N, n_head]
        struct ggml_tensor * attn = ggml_mul_mat(ctx, K, Q);
        attn = ggml_scale(ctx, attn, 1.0f / sqrtf((float)n_embd_head));

        // BIDIRECTIONAL: no causal mask
        attn = ggml_soft_max(ctx, attn);

        // Weighted sum: V^T @ attn → [n_embd_head, N, n_head]
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, V, attn);

        // Merge heads: [n_embd_head, N, n_head] → [n_embd_head, n_head, N] → [n_embd, N]
        attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, N);

        // Output projection
        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN (SwiGLU) ───────────────────────────────────────
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

    // ── Final norm + logits ────────────────────────────────────
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

// ── Build full graph WITH named K,V for cache extraction ────────
// Same as diffuse_build_graph but marks K,V tensors as outputs so
// they can be extracted after execution to populate the cache.
static struct ggml_cgraph * diffuse_build_graph_extractable(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_tokens) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int N           = n_tokens;

    // Extra nodes for the K,V output markers
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx,
            (size_t)(4096 * n_layer + n_layer * 4), false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, tokens, N * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    {
        int32_t * pos_data = (int32_t *)inp_pos->data;
        for (int i = 0; i < N; i++) pos_data[i] = i;
    }

    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K = ggml_add(ctx, K, ensure_f32(ctx, layer.bk));
        if (layer.bv) V = ggml_add(ctx, V, ensure_f32(ctx, layer.bv));

        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat each KV head n_rep times (grouped, not interleaved)
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K = ggml_reshape_4d(ctx, K, n_embd_head, 1, n_head_kv, N);
            K = ggml_repeat(ctx, K,
                    ggml_new_tensor_4d(ctx, K->type, n_embd_head, n_rep, n_head_kv, N));
            K = ggml_reshape_3d(ctx, K, n_embd_head, n_head, N);

            V = ggml_reshape_4d(ctx, V, n_embd_head, 1, n_head_kv, N);
            V = ggml_repeat(ctx, V,
                    ggml_new_tensor_4d(ctx, V->type, n_embd_head, n_rep, n_head_kv, N));
            V = ggml_reshape_3d(ctx, V, n_embd_head, n_head, N);
        }

        // ── Name K,V for cache extraction (BEFORE permute) ──────
        // Shape at this point: [n_embd_head, n_head, N]
        {
            char name_buf[32];
            snprintf(name_buf, sizeof(name_buf), "Kc.%02d", il);
            ggml_set_name(K, name_buf);
            ggml_set_output(K);

            snprintf(name_buf, sizeof(name_buf), "Vc.%02d", il);
            ggml_set_name(V, name_buf);
            ggml_set_output(V);
        }

        // Permute for attention (same as original)
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

        struct ggml_tensor * attn = ggml_mul_mat(ctx, K, Q);
        attn = ggml_scale(ctx, attn, 1.0f / sqrtf((float)n_embd_head));
        attn = ggml_soft_max(ctx, attn);

        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, V, attn);
        attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, N);

        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

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

// ── Full forward pass with cache extraction ─────────────────────
bool diffuse_forward_full(diffuse_context * ctx,
                          const int32_t * tokens, int n_tokens,
                          float * logits_out,
                          diffuse_step_cache * cache) {
    const auto & hp = ctx->model->hparams;

    // Buffer size: same as original + extra for named K,V output tensors
    size_t per_layer = (size_t)n_tokens * hp.n_embd * sizeof(float) * 10
                     + (size_t)n_tokens * hp.n_ff   * sizeof(float) * 3
                     + (size_t)n_tokens * n_tokens * hp.n_head * sizeof(float) * 2
                     + hp.n_embd * sizeof(float) * 2;
    size_t buf_size = per_layer * hp.n_layer;
    buf_size += (size_t)n_tokens * hp.n_vocab * sizeof(float) * 2;
    buf_size += 256ull * 1024 * 1024;
    buf_size = (size_t)(buf_size * 1.5);

    struct ggml_init_params cparams = {
        /*.mem_size   = */ buf_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    struct ggml_context * ctx_compute = ggml_init(cparams);
    if (!ctx_compute) {
        DIFFUSE_LOG("failed to allocate compute context (%zu MB)", buf_size / (1024*1024));
        return false;
    }

    struct ggml_cgraph * gf = (cache != nullptr)
        ? diffuse_build_graph_extractable(ctx, ctx_compute, tokens, n_tokens)
        : diffuse_build_graph(ctx, ctx_compute, tokens, n_tokens);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        DIFFUSE_LOG("graph compute failed with status %d", (int)status);
        ggml_free(ctx_compute);
        return false;
    }

    // Extract logits
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));

    // Extract K,V into cache
    if (cache) {
        const int n_layer = (int)hp.n_layer;
        size_t kv_bytes = cache->pos_stride() * n_tokens * sizeof(float);
        char name_buf[32];

        for (int il = 0; il < n_layer; il++) {
            snprintf(name_buf, sizeof(name_buf), "Kc.%02d", il);
            struct ggml_tensor * K_t = ggml_graph_get_tensor(gf, name_buf);
            snprintf(name_buf, sizeof(name_buf), "Vc.%02d", il);
            struct ggml_tensor * V_t = ggml_graph_get_tensor(gf, name_buf);

            if (K_t && V_t) {
                memcpy(cache->K[il].data(), K_t->data, kv_bytes);
                memcpy(cache->V[il].data(), V_t->data, kv_bytes);
            } else {
                DIFFUSE_LOG("WARNING: could not extract K/V for layer %d", il);
            }
        }
    }

    ggml_free(ctx_compute);
    return true;
}

// ── Build CACHED graph for active positions only ────────────────
//
// Architecture:
//   - Compute Q, K, V projections ONLY for active_tokens
//   - For attention: K_full = concat(K_cached, K_active), V_full = concat(V_cached, V_active)
//   - K_cached contains the cached K,V for INACTIVE positions (reordered)
//   - The attention is Q_active × K_full, giving scores [n_total, n_active, n_head]
//   - FFN and logits computed only for n_active positions
//
// Position ordering:
//   K_full[0..n_cached-1] = cached positions (in order of cached_positions[])
//   K_full[n_cached..n_total-1] = active positions (in order of active_positions[])
//   RoPE uses the ORIGINAL absolute positions.

struct ggml_cgraph * diffuse_build_graph_cached(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * active_tokens,
        const int32_t * active_pos_indices,
        int n_active,
        int n_total,
        diffuse_step_cache * cache,
        const std::vector<int> & cached_positions) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int n_cached    = n_total - n_active;
    const size_t kv_stride = cache->pos_stride();  // n_embd_head * n_head

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx,
            (size_t)(4096 * n_layer + n_layer * 8 + 256), false);

    // ── Active position inputs ───────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_active);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, active_tokens, n_active * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_active);
    ggml_set_name(inp_pos, "inp_pos_active");
    ggml_set_input(inp_pos);
    memcpy(inp_pos->data, active_pos_indices, n_active * sizeof(int32_t));

    // We also need position indices for the cached positions (for V permutation ordering)
    // — not needed for RoPE since cached K,V already have RoPE applied

    // ── Embedding for active positions ───────────────────────────
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

        // QKV projections for active positions only
        struct ggml_tensor * Q_active = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K_active = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V_active = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q_active = ggml_add(ctx, Q_active, ensure_f32(ctx, layer.bq));
        if (layer.bk) K_active = ggml_add(ctx, K_active, ensure_f32(ctx, layer.bk));
        if (layer.bv) V_active = ggml_add(ctx, V_active, ensure_f32(ctx, layer.bv));

        // Reshape: [n_embd, n_active] → [n_embd_head, n_head(_kv), n_active]
        Q_active = ggml_reshape_3d(ctx, Q_active, n_embd_head, n_head,    n_active);
        K_active = ggml_reshape_3d(ctx, K_active, n_embd_head, n_head_kv, n_active);
        V_active = ggml_reshape_3d(ctx, V_active, n_embd_head, n_head_kv, n_active);

        // RoPE on Q and K with ORIGINAL position indices
        Q_active = ggml_rope_ext(ctx, Q_active, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_active = ggml_rope_ext(ctx, K_active, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat each KV head n_rep times (grouped, not interleaved)
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K_active = ggml_reshape_4d(ctx, K_active, n_embd_head, 1, n_head_kv, n_active);
            K_active = ggml_repeat(ctx, K_active,
                    ggml_new_tensor_4d(ctx, K_active->type, n_embd_head, n_rep, n_head_kv, n_active));
            K_active = ggml_reshape_3d(ctx, K_active, n_embd_head, n_head, n_active);

            V_active = ggml_reshape_4d(ctx, V_active, n_embd_head, 1, n_head_kv, n_active);
            V_active = ggml_repeat(ctx, V_active,
                    ggml_new_tensor_4d(ctx, V_active->type, n_embd_head, n_rep, n_head_kv, n_active));
            V_active = ggml_reshape_3d(ctx, V_active, n_embd_head, n_head, n_active);
        }

        // ── Name K_active, V_active for extraction ──────────────
        {
            char name_buf[32];
            snprintf(name_buf, sizeof(name_buf), "Ka.%02d", il);
            ggml_set_name(K_active, name_buf);
            ggml_set_output(K_active);

            snprintf(name_buf, sizeof(name_buf), "Va.%02d", il);
            ggml_set_name(V_active, name_buf);
            ggml_set_output(V_active);
        }

        // ── Load cached K,V for inactive positions ──────────────
        // Shape: [n_embd_head, n_head, n_cached]
        struct ggml_tensor * K_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                n_embd_head, n_head, n_cached);
        ggml_set_input(K_cached);

        struct ggml_tensor * V_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                n_embd_head, n_head, n_cached);
        ggml_set_input(V_cached);

        // Fill from cache: gather cached positions into contiguous tensor
        {
            float * K_dst = (float *)K_cached->data;
            float * V_dst = (float *)V_cached->data;
            for (int c = 0; c < n_cached; c++) {
                int orig_pos = cached_positions[c];
                memcpy(K_dst + c * kv_stride,
                       cache->K[il].data() + orig_pos * kv_stride,
                       kv_stride * sizeof(float));
                memcpy(V_dst + c * kv_stride,
                       cache->V[il].data() + orig_pos * kv_stride,
                       kv_stride * sizeof(float));
            }
        }

        // ── Concatenate: K_full = [K_cached | K_active] ─────────
        // dim=2 is the sequence/position dimension
        struct ggml_tensor * K_full = ggml_concat(ctx, K_cached, K_active, 2);
        struct ggml_tensor * V_full = ggml_concat(ctx, V_cached, V_active, 2);
        // K_full shape: [n_embd_head, n_head, n_total]

        // ── Permute for attention ────────────────────────────────
        // Q: [n_embd_head, n_head, n_active] → [n_embd_head, n_active, n_head]
        Q_active = ggml_permute(ctx, Q_active, 0, 2, 1, 3);
        // K: [n_embd_head, n_head, n_total] → [n_embd_head, n_total, n_head]
        K_full = ggml_permute(ctx, K_full, 0, 2, 1, 3);
        // V: [n_embd_head, n_head, n_total] → [n_total, n_embd_head, n_head]
        V_full = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

        // ── Attention scores: K_full^T @ Q_active ────────────────
        // K_full: [n_embd_head, n_total, n_head]
        // Q_active: [n_embd_head, n_active, n_head]
        // Result: [n_total, n_active, n_head]
        struct ggml_tensor * attn = ggml_mul_mat(ctx, K_full, Q_active);
        attn = ggml_scale(ctx, attn, 1.0f / sqrtf((float)n_embd_head));
        attn = ggml_soft_max(ctx, attn);

        // ── Weighted sum: V_full^T @ attn ────────────────────────
        // V_full: [n_total, n_embd_head, n_head]
        // attn: [n_total, n_active, n_head]
        // Result: [n_embd_head, n_active, n_head]
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, V_full, attn);

        // Merge heads: [n_embd_head, n_active, n_head] → [n_embd, n_active]
        attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        attn_out = ggml_cont(ctx, attn_out);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, n_active);

        // Output projection
        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN (SwiGLU) — only for active positions ────────────
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

    // ── Final norm + logits (active positions only) ──────────────
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

// ── Cached forward pass execution ───────────────────────────────
bool diffuse_forward_cached(
        diffuse_context * ctx,
        const int32_t * active_tokens,
        const int32_t * active_pos_indices,
        int n_active,
        int n_total,
        diffuse_step_cache * cache,
        const std::vector<int> & cached_positions,
        const std::vector<int> & active_positions,
        float * logits_out) {

    const auto & hp = ctx->model->hparams;
    const int n_cached = n_total - n_active;

    // Compute buffer: sized for active + cached attention
    size_t per_layer = (size_t)n_active * hp.n_embd * sizeof(float) * 10
                     + (size_t)n_active * hp.n_ff   * sizeof(float) * 3
                     + (size_t)n_active * n_total * hp.n_head * sizeof(float) * 2
                     + (size_t)n_total * hp.n_embd_head() * hp.n_head * sizeof(float) * 4
                     + hp.n_embd * sizeof(float) * 2;
    size_t buf_size = per_layer * hp.n_layer;
    buf_size += (size_t)n_active * hp.n_vocab * sizeof(float) * 2;
    buf_size += 256ull * 1024 * 1024;
    buf_size = (size_t)(buf_size * 1.5);

    struct ggml_init_params cparams = {
        /*.mem_size   = */ buf_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    struct ggml_context * ctx_compute = ggml_init(cparams);
    if (!ctx_compute) {
        DIFFUSE_LOG("cached: failed to alloc compute ctx (%zu MB)", buf_size / (1024*1024));
        return false;
    }

    struct ggml_cgraph * gf = diffuse_build_graph_cached(
            ctx, ctx_compute,
            active_tokens, active_pos_indices,
            n_active, n_total,
            cache, cached_positions);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        DIFFUSE_LOG("cached graph compute failed with status %d", (int)status);
        ggml_free(ctx_compute);
        return false;
    }

    // Extract logits for active positions
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found in cached graph");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_active * hp.n_vocab * sizeof(float));

    // Update cache: store K_active, V_active for active positions
    {
        char name_buf[32];
        for (int il = 0; il < (int)hp.n_layer; il++) {
            snprintf(name_buf, sizeof(name_buf), "Ka.%02d", il);
            struct ggml_tensor * K_t = ggml_graph_get_tensor(gf, name_buf);
            snprintf(name_buf, sizeof(name_buf), "Va.%02d", il);
            struct ggml_tensor * V_t = ggml_graph_get_tensor(gf, name_buf);

            if (K_t && V_t) {
                cache->update_kv(il, (const float *)K_t->data,
                                     (const float *)V_t->data,
                                     active_positions);
            }
        }
    }

    ggml_free(ctx_compute);
    return true;
}

// ── Original forward pass (unchanged, for backward compat) ──────
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out) {
    return diffuse_forward_full(ctx, tokens, n_tokens, logits_out, nullptr);
}

// ── Context management ─────────────────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads) {
    auto * ctx = new diffuse_context();
    ctx->model     = model;
    ctx->n_ctx     = n_ctx;
    ctx->n_threads = n_threads;
    return ctx;
}

void diffuse_context_free(diffuse_context * ctx) {
    if (!ctx) return;
    if (ctx->buf) ggml_backend_buffer_free(ctx->buf);
    if (ctx->ctx) ggml_free(ctx->ctx);
    delete ctx;
}
