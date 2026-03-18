#include "diffuse-model.h"

// ── GGUF metadata key helpers ──────────────────────────────────
static int64_t find_key(const struct gguf_context * gctx, const char * key) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) {
        DIFFUSE_DIE("missing GGUF metadata key: %s", key);
    }
    return id;
}

static uint32_t get_u32(const struct gguf_context * gctx, const char * key) {
    return gguf_get_val_u32(gctx, find_key(gctx, key));
}

static float get_f32(const struct gguf_context * gctx, const char * key, float def) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return def;
    return gguf_get_val_f32(gctx, id);
}

// ── Tensor lookup helper ───────────────────────────────────────
static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        DIFFUSE_DIE("tensor not found: %s", name);
    }
    return t;
}

// ── Load model from GGUF ───────────────────────────────────────
diffuse_model * diffuse_model_load_impl(const std::string & path, int n_threads) {
    DIFFUSE_LOG("loading model from %s", path.c_str());

    // Open GGUF file — gguf_init_from_file allocates a ggml_context with tensors
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gparams = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &meta_ctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), gparams);
    if (!gctx) {
        DIFFUSE_DIE("failed to open GGUF file: %s", path.c_str());
    }

    // Parse hyperparameters (standard GGUF key names)
    auto * model = new diffuse_model();
    auto & hp = model->hparams;

    hp.n_layer       = get_u32(gctx, "diffuse.block_count");
    hp.n_head        = get_u32(gctx, "diffuse.attention.head_count");
    hp.n_head_kv     = get_u32(gctx, "diffuse.attention.head_count_kv");
    hp.n_embd        = get_u32(gctx, "diffuse.embedding_length");
    hp.n_ff          = get_u32(gctx, "diffuse.feed_forward_length");
    hp.n_ctx_max     = get_u32(gctx, "diffuse.context_length");
    hp.n_vocab       = get_u32(gctx, "diffuse.vocab_size");
    hp.mask_token_id = get_u32(gctx, "diffuse.mask_token_id");
    hp.rope_theta    = get_f32(gctx, "diffuse.rope.freq_base", 500000.0f);
    hp.rms_norm_eps  = get_f32(gctx, "diffuse.attention.layer_norm_rms_epsilon", 1e-5f);

    DIFFUSE_LOG("  arch: LLaDA (Llama backbone)");
    DIFFUSE_LOG("  n_vocab=%u, n_embd=%u, n_head=%u, n_layer=%u, n_ff=%u",
                hp.n_vocab, hp.n_embd, hp.n_head, hp.n_layer, hp.n_ff);
    DIFFUSE_LOG("  mask_token_id=%u, rope_theta=%.0f", hp.mask_token_id, hp.rope_theta);

    // The weight context is the one that gguf_init populated
    model->ctx = meta_ctx;

    // Map tensors to model struct
    model->tok_embd    = get_tensor(meta_ctx, "token_embd.weight");
    model->output_norm = get_tensor(meta_ctx, "output_norm.weight");

    // lm_head — may or may not be present (tied embeddings)
    struct ggml_tensor * lm_head = ggml_get_tensor(meta_ctx, "output.weight");
    model->output = lm_head ? lm_head : model->tok_embd;

    // Layers
    model->layers.resize(hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto & l = model->layers[i];
        l.attn_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_norm.weight", i).c_str());
        l.wq       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_q.weight", i).c_str());
        l.wk       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_k.weight", i).c_str());
        l.wv       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_v.weight", i).c_str());
        l.wo       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_output.weight", i).c_str());
        l.ffn_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_norm.weight", i).c_str());
        l.ffn_gate = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_gate.weight", i).c_str());
        l.ffn_up   = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_up.weight", i).c_str());
        l.ffn_down = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_down.weight", i).c_str());
    }

    DIFFUSE_LOG("model loaded: %lld tensors",
                (long long)gguf_get_n_tensors(gctx));
    gguf_free(gctx);  // frees metadata, but ggml_context (weights) stays alive

    return model;
}

void diffuse_model_free_impl(diffuse_model * model) {
    if (!model) return;
    if (model->buf)     ggml_backend_buffer_free(model->buf);
    if (model->ctx)     ggml_free(model->ctx);
    if (model->backend) ggml_backend_free(model->backend);
    delete model;
}

// ── Public API wrappers ────────────────────────────────────────
diffuse_model * diffuse_model_load(const std::string & path, int n_threads) {
    return diffuse_model_load_impl(path, n_threads);
}

void diffuse_model_free(diffuse_model * model) {
    diffuse_model_free_impl(model);
}

const diffuse_hparams & diffuse_model_hparams(const diffuse_model * model) {
    return model->hparams;
}
