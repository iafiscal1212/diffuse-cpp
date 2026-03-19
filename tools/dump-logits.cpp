// dump-logits: Run forward pass and save raw logits to binary file.
// Used for cross-validation against PyTorch.
//
// Usage: dump-logits -m model.gguf --tokens 1,2,3,4 -o logits.bin -t 4
// Output: binary file with float32 logits [n_tokens × n_vocab]

#include "diffuse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

static std::vector<int32_t> parse_tokens(const char * str) {
    std::vector<int32_t> tokens;
    const char * p = str;
    while (*p) {
        tokens.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return tokens;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_path = "logits.bin";
    std::vector<int32_t> tokens;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        }
    }

    if (model_path.empty() || tokens.empty()) {
        fprintf(stderr, "Usage: dump-logits -m MODEL.gguf --tokens 1,2,3 [-o logits.bin] [-t threads]\n");
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    diffuse_model * model = diffuse_model_load(model_path, n_threads);
    if (!model) return 1;

    const auto & hp = diffuse_model_hparams(model);
    int n_tokens = (int)tokens.size();
    fprintf(stderr, "Forward pass: %d tokens, %d threads\n", n_tokens, n_threads);

    diffuse_context * ctx = diffuse_context_new(model, n_tokens, n_threads);

    std::vector<float> logits(n_tokens * hp.n_vocab);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = diffuse_forward(ctx, tokens.data(), n_tokens, logits.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!ok) {
        fprintf(stderr, "Forward pass failed\n");
        return 1;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Forward pass: %.1f ms\n", ms);

    // Save logits to binary file
    FILE * f = fopen(output_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", output_path.c_str());
        return 1;
    }

    // Header: n_tokens (int32), n_vocab (int32)
    int32_t header[2] = { n_tokens, (int32_t)hp.n_vocab };
    fwrite(header, sizeof(int32_t), 2, f);

    // Data: float32 logits [n_tokens * n_vocab]
    fwrite(logits.data(), sizeof(float), logits.size(), f);
    fclose(f);

    size_t file_size = sizeof(int32_t) * 2 + sizeof(float) * logits.size();
    fprintf(stderr, "Logits saved: %s (%.1f MB, %d × %u)\n",
            output_path.c_str(), file_size / (1024.0 * 1024.0),
            n_tokens, hp.n_vocab);

    // Print top-5 per position for quick check
    fprintf(stderr, "\nTop-5 predictions:\n");
    for (int i = 0; i < n_tokens && i < 8; i++) {
        const float * row = logits.data() + (size_t)i * hp.n_vocab;

        // Find top 5
        std::vector<std::pair<float, int>> scores;
        for (uint32_t v = 0; v < hp.n_vocab; v++) {
            scores.push_back({row[v], (int)v});
        }
        std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        fprintf(stderr, "  pos %d (in=%d): ", i, tokens[i]);
        for (int k = 0; k < 5; k++) {
            fprintf(stderr, "%d(%.2f) ", scores[k].second, scores[k].first);
        }
        fprintf(stderr, "\n");
    }

    diffuse_context_free(ctx);
    diffuse_model_free(model);
    return 0;
}
