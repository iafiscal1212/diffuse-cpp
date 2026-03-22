#include "diffuse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -m PATH     Model file (GGUF)\n");
    fprintf(stderr, "  -p TEXT     Prompt text\n");
    fprintf(stderr, "  -n INT      Tokens to generate (default: 128)\n");
    fprintf(stderr, "  -s INT      Diffusion steps (default: 32)\n");
    fprintf(stderr, "  -t INT      Threads (default: 4)\n");
    fprintf(stderr, "  --temp F    Temperature (default: 0 = argmax)\n");
    fprintf(stderr, "  --seed INT  Random seed (default: 42)\n");
    fprintf(stderr, "  --schedule  cosine|linear (default: cosine)\n");
    fprintf(stderr, "  --remasking low_confidence|random|entropy_exit|maskgit_plus|topk_margin (default: low_confidence)\n");
    fprintf(stderr, "  --entropy-threshold F  Entropy threshold for entropy_exit (default: 1.5)\n");
    fprintf(stderr, "  --cache-refresh INT   Force full forward every N steps (default: 0 = never)\n");
    fprintf(stderr, "  --cache-keep-active INT  Keep recently-changed positions active N extra steps (default: 0)\n");
    fprintf(stderr, "\nNote: Tokenization is currently external. Use --tokens to pass\n");
    fprintf(stderr, "      pre-tokenized input as comma-separated IDs.\n");
    fprintf(stderr, "  --tokens IDs  Comma-separated token IDs (bypasses prompt)\n");
}

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
    std::string prompt;
    std::vector<int32_t> input_tokens;
    int n_generate = 128;
    int n_steps    = 32;
    int n_threads  = 4;
    float temperature = 0.0f;
    float entropy_threshold = 1.5f;
    uint32_t seed  = 42;
    diffuse_schedule schedule = diffuse_schedule::COSINE;
    diffuse_remasking remasking = diffuse_remasking::LOW_CONFIDENCE;
    bool use_cache = true;
    int cache_refresh = 0;
    int cache_keep_active = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_generate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            input_tokens = parse_tokens(argv[++i]);
        } else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "linear") == 0) schedule = diffuse_schedule::LINEAR;
        } else if (strcmp(argv[i], "--entropy-threshold") == 0 && i + 1 < argc) {
            entropy_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "--remasking") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "random") == 0) remasking = diffuse_remasking::RANDOM;
            else if (strcmp(argv[i], "entropy_exit") == 0) remasking = diffuse_remasking::ENTROPY_EXIT;
            else if (strcmp(argv[i], "maskgit_plus") == 0) remasking = diffuse_remasking::MASKGIT_PLUS;
            else if (strcmp(argv[i], "topk_margin") == 0) remasking = diffuse_remasking::TOPK_MARGIN;
        } else if (strcmp(argv[i], "--no-cache") == 0) {
            use_cache = false;
        } else if (strcmp(argv[i], "--cache-refresh") == 0 && i + 1 < argc) {
            cache_refresh = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-keep-active") == 0 && i + 1 < argc) {
            cache_keep_active = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: model path required (-m)\n");
        print_usage(argv[0]);
        return 1;
    }
    if (input_tokens.empty() && prompt.empty()) {
        fprintf(stderr, "Error: prompt (-p) or tokens (--tokens) required\n");
        return 1;
    }

    // Native tokenizer not integrated; use --tokens with pre-tokenized input
    if (input_tokens.empty()) {
        fprintf(stderr, "Warning: native tokenizer not yet integrated. "
                        "Use --tokens with pre-tokenized input.\n");
        fprintf(stderr, "Example: python -c \"from transformers import AutoTokenizer; "
                        "t=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct'); "
                        "print(','.join(map(str,t.encode('%s'))))\"\n", prompt.c_str());
        return 1;
    }

    // Load model
    fprintf(stderr, "Loading model...\n");
    diffuse_model * model = diffuse_model_load(model_path, n_threads);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = diffuse_model_hparams(model);
    int n_ctx = (int)input_tokens.size() + n_generate;
    diffuse_context * ctx = diffuse_context_new(model, n_ctx, n_threads);

    // Setup sampler params
    diffuse_sampler_params sparams;
    sparams.n_steps     = n_steps;
    sparams.temperature = temperature;
    sparams.seed        = seed;
    sparams.schedule    = schedule;
    sparams.remasking   = remasking;
    sparams.entropy_threshold = entropy_threshold;
    sparams.use_cache = use_cache;
    sparams.cache_refresh = cache_refresh;
    sparams.cache_keep_active = cache_keep_active;

    // Generate
    fprintf(stderr, "Generating %d tokens with %d diffusion steps...\n", n_generate, n_steps);

    auto result = diffuse_generate(ctx, input_tokens, n_generate, sparams,
        [](int step, int total, const std::vector<int32_t> & tokens) {
            fprintf(stderr, "\r  step %d/%d", step, total);
        });

    fprintf(stderr, "\n\nGenerated token IDs:\n");
    for (size_t i = 0; i < result.size(); i++) {
        if (i > 0) printf(",");
        printf("%d", result[i]);
    }
    printf("\n");

    fprintf(stderr, "\nDecode with: python -c \"from transformers import AutoTokenizer; "
                    "t=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct'); "
                    "print(t.decode([...]))\"\n");

    // Cleanup
    diffuse_context_free(ctx);
    diffuse_model_free(model);

    return 0;
}
