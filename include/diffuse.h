#pragma once

// diffuse-cpp public API

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

// Forward declarations
struct diffuse_model;
struct diffuse_context;

// ── Hyperparameters ────────────────────────────────────────────
struct diffuse_hparams {
    uint32_t n_vocab      = 0;
    uint32_t n_embd       = 0;   // hidden_size
    uint32_t n_head       = 0;
    uint32_t n_head_kv    = 0;   // for GQA; equals n_head for MHA
    uint32_t n_layer      = 0;
    uint32_t n_ff         = 0;   // intermediate_size
    uint32_t n_ctx_max    = 0;   // max sequence length
    float    rope_theta   = 500000.0f;
    float    rms_norm_eps = 1e-5f;
    uint32_t mask_token_id = 0;

    // Derived
    uint32_t n_embd_head() const { return n_head > 0 ? n_embd / n_head : 0; }
};

// ── Sampler parameters ─────────────────────────────────────────
enum class diffuse_schedule {
    COSINE,
    LINEAR,
};

enum class diffuse_remasking {
    LOW_CONFIDENCE,
    RANDOM,
    ENTROPY_EXIT,   // Unmask all low-entropy tokens early (semantic scheduling)
    MASKGIT_PLUS,   // Dream: unmask highest top-1 confidence (similar to LOW_CONFIDENCE)
    TOPK_MARGIN,    // Dream: unmask by margin between top-1 and top-2 logits
};

struct diffuse_sampler_params {
    int      n_steps     = 32;
    float    temperature = 0.0f;  // 0 = argmax
    diffuse_schedule   schedule   = diffuse_schedule::COSINE;
    diffuse_remasking  remasking  = diffuse_remasking::LOW_CONFIDENCE;
    uint32_t seed       = 42;
    float    entropy_threshold = 1.5f;  // For ENTROPY_EXIT: unmask tokens below this
    bool     use_cache   = true;        // Inter-step KV cache (--no-cache to disable)
    int      cache_refresh    = 0;      // Force full forward every N steps (0 = never)
    int      cache_keep_active = 0;     // Keep recently-changed positions active N extra steps
};

// ── Generation parameters ──────────────────────────────────────
struct diffuse_params {
    std::string model_path;
    std::string prompt;
    int         n_generate  = 128;  // tokens to generate
    int         n_threads   = 4;
    diffuse_sampler_params sampler;
};

// ── Token callback (called after each diffusion step) ──────────
using diffuse_step_callback = std::function<void(
    int step, int total_steps, const std::vector<int32_t>& tokens)>;

// ── Model API ──────────────────────────────────────────────────
diffuse_model * diffuse_model_load(const std::string & path, int n_threads);
void            diffuse_model_free(diffuse_model * model);
const diffuse_hparams & diffuse_model_hparams(const diffuse_model * model);

// ── Context (holds compute buffers) ────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads);
void              diffuse_context_free(diffuse_context * ctx);

// ── Forward pass ───────────────────────────────────────────────
// tokens: [n_tokens], logits_out: [n_tokens * n_vocab]
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out);

// ── Generation (full diffusion loop) ───────────────────────────
std::vector<int32_t> diffuse_generate(
    diffuse_context * ctx,
    const std::vector<int32_t> & prompt_tokens,
    int n_generate,
    const diffuse_sampler_params & params,
    diffuse_step_callback callback = nullptr);

// ═══════════════════════════════════════════════════════════════
// ── Autoregressive API ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

// ── AR sampler parameters ─────────────────────────────────────
struct ar_sampler_params {
    float    temperature    = 0.0f;   // 0 = greedy/argmax
    float    top_p          = 0.9f;   // nucleus sampling
    int      top_k          = 40;     // top-k (0 = disabled)
    float    repeat_penalty = 1.1f;   // repetition penalty
    int      repeat_last_n  = 64;     // window for repeat penalty
    uint32_t seed           = 42;
};

// ── AR token callback (return false to stop) ──────────────────
using ar_token_callback = std::function<bool(int32_t token, int pos)>;

// ── AR generation ─────────────────────────────────────────────
std::vector<int32_t> ar_generate(
    diffuse_context * ctx,
    const std::vector<int32_t> & prompt_tokens,
    int max_tokens,
    const ar_sampler_params & params,
    ar_token_callback callback = nullptr);

// ═══════════════════════════════════════════════════════════════
// ── Speculative Decoding API ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

struct ar_spec_params {
    int K = 5;   // speculative lookahead (draft tokens per round)
};

struct ar_spec_stats {
    int total_draft_tokens  = 0;
    int total_accepted      = 0;
    int total_generated     = 0;
    int total_target_batches = 0;
    int total_target_decodes = 0;
    int total_fast_rejects   = 0;
    double prefill_ms = 0;
    double decode_ms  = 0;

    float acceptance_rate() const {
        return total_draft_tokens > 0
            ? (float)total_accepted / total_draft_tokens : 0.0f;
    }
    float tokens_per_target_call() const {
        int calls = total_target_batches + total_target_decodes;
        return calls > 0 ? (float)total_generated / calls : 1.0f;
    }
};

std::vector<int32_t> ar_speculative_generate(
    diffuse_context * target_ctx,
    diffuse_context * draft_ctx,
    const std::vector<int32_t> & prompt_tokens,
    int max_tokens,
    const ar_spec_params & params,
    ar_token_callback callback = nullptr,
    ar_spec_stats * stats_out = nullptr);

// end of diffuse.h
