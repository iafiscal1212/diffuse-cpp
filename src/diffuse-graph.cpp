#include "diffuse-graph.h"

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

        // Reshape for multi-head: [n_embd_head, n_head, N]
        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        // RoPE on Q and K (NEOX style = non-interleaved, like LLaDA/OLMo)
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat K, V heads if n_head_kv < n_head
        if (n_head_kv < n_head) {
            K = ggml_repeat(ctx, K,
                    ggml_new_tensor_3d(ctx, K->type, n_embd_head, n_head, N));
            V = ggml_repeat(ctx, V,
                    ggml_new_tensor_3d(ctx, V->type, n_embd_head, n_head, N));
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

// ── Forward pass execution ─────────────────────────────────────
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out) {
    const auto & hp = ctx->model->hparams;

    // Compute buffer: all intermediate tensors for forward pass
    // Per layer: ~10 embd-sized + 3 ff-sized + 2 attn-score matrices
    size_t per_layer = (size_t)n_tokens * hp.n_embd * sizeof(float) * 10
                     + (size_t)n_tokens * hp.n_ff   * sizeof(float) * 3
                     + (size_t)n_tokens * n_tokens * hp.n_head * sizeof(float) * 2
                     + hp.n_embd * sizeof(float) * 2;  // norm weight casts
    size_t buf_size = per_layer * hp.n_layer;
    buf_size += (size_t)n_tokens * hp.n_vocab * sizeof(float) * 2;  // logits
    buf_size += 256ull * 1024 * 1024;  // graph nodes + execution plan overhead
    buf_size = (size_t)(buf_size * 1.2);  // 20% safety margin

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

    struct ggml_cgraph * gf = diffuse_build_graph(ctx, ctx_compute, tokens, n_tokens);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        DIFFUSE_LOG("graph compute failed with status %d", (int)status);
        ggml_free(ctx_compute);
        return false;
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found in graph");
        ggml_free(ctx_compute);
        return false;
    }

    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));
    ggml_free(ctx_compute);
    return true;
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
