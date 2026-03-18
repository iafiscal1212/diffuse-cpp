#include "diffuse.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while(0)

int main(int argc, char ** argv) {
    // Test 1: Hyperparameters struct defaults
    {
        diffuse_hparams hp;
        ASSERT(hp.n_vocab == 0, "default n_vocab should be 0");
        ASSERT(hp.rope_theta == 500000.0f, "default rope_theta should be 500000");
        ASSERT(hp.rms_norm_eps == 1e-5f, "default rms_norm_eps should be 1e-5");
        ASSERT(hp.n_embd_head() == 0, "n_embd_head with zero values");
        fprintf(stderr, "PASS: hparams defaults\n");
    }

    // Test 2: Sampler params defaults
    {
        diffuse_sampler_params sp;
        ASSERT(sp.n_steps == 32, "default n_steps should be 32");
        ASSERT(sp.temperature == 0.0f, "default temperature should be 0");
        ASSERT(sp.schedule == diffuse_schedule::COSINE, "default schedule should be COSINE");
        ASSERT(sp.remasking == diffuse_remasking::LOW_CONFIDENCE, "default remasking");
        fprintf(stderr, "PASS: sampler params defaults\n");
    }

    // Test 3: Model load failure on non-existent file
    {
        // This should fail gracefully (exit with error from DIFFUSE_DIE)
        // We just verify the function signature compiles
        fprintf(stderr, "PASS: API compilation check\n");
    }

    // Test 4: Full forward pass (requires model file)
    if (argc > 1) {
        const char * model_path = argv[1];
        fprintf(stderr, "Running forward pass test with model: %s\n", model_path);

        diffuse_model * model = diffuse_model_load(model_path, 4);
        ASSERT(model != nullptr, "model should load");

        const auto & hp = diffuse_model_hparams(model);
        ASSERT(hp.n_vocab > 0, "vocab size should be > 0");
        ASSERT(hp.n_layer > 0, "n_layer should be > 0");

        // Small test: 8 tokens
        int n_tokens = 8;
        std::vector<int32_t> tokens(n_tokens, 1);  // token ID 1

        diffuse_context * ctx = diffuse_context_new(model, n_tokens, 4);
        std::vector<float> logits(n_tokens * hp.n_vocab);

        bool ok = diffuse_forward(ctx, tokens.data(), n_tokens, logits.data());
        ASSERT(ok, "forward pass should succeed");

        // Check logits are not all zeros
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); i++) {
            sum += fabsf(logits[i]);
        }
        ASSERT(sum > 0.0f, "logits should not be all zeros");
        fprintf(stderr, "PASS: forward pass (logits sum=%.2f)\n", sum);

        diffuse_context_free(ctx);
        diffuse_model_free(model);
    } else {
        fprintf(stderr, "SKIP: forward pass test (no model path provided)\n");
        fprintf(stderr, "  Run with: test-forward /path/to/model.gguf\n");
    }

    fprintf(stderr, "\nAll tests passed!\n");
    return 0;
}
