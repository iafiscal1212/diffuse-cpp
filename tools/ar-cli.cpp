#include "diffuse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\ndiffuse-cpp autoregressive inference engine\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -m PATH      Model file (GGUF, llama.cpp or diffuse-cpp format)\n");
    fprintf(stderr, "  -p TEXT      Prompt text (requires external tokenization)\n");
    fprintf(stderr, "  -n INT       Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -t INT       Threads (default: 4)\n");
    fprintf(stderr, "  --temp F     Temperature (default: 0 = greedy)\n");
    fprintf(stderr, "  --top-p F    Nucleus sampling threshold (default: 0.9)\n");
    fprintf(stderr, "  --top-k INT  Top-K sampling (default: 40, 0 = disabled)\n");
    fprintf(stderr, "  --repeat-penalty F   Repetition penalty (default: 1.1)\n");
    fprintf(stderr, "  --repeat-last-n INT  Repeat penalty window (default: 64)\n");
    fprintf(stderr, "  --seed INT   Random seed (default: 42)\n");
    fprintf(stderr, "  --tokens IDs Comma-separated pre-tokenized input\n");
    fprintf(stderr, "  --bind-cores  Bind threads to CPU cores (reduces jitter)\n");
    fprintf(stderr, "\nSpeculative decoding:\n");
    fprintf(stderr, "  --draft PATH  Draft model GGUF (enables speculative decoding)\n");
    fprintf(stderr, "  --spec-k INT  Speculative lookahead (default: 4)\n");
    fprintf(stderr, "  --draft-threads INT  Threads for draft model (default: 4)\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s -m thrombia-32b-q4.gguf --tokens 151644,8948,198,... -n 512 -t 12\n", prog);
    fprintf(stderr, "  %s -m thrombia-32b-q4.gguf --draft qwen2.5-0.5b-q4.gguf --spec-k 5 -t 12\n", prog);
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
    std::string draft_path;
    std::string prompt;
    std::vector<int32_t> input_tokens;
    int max_tokens = 256;
    int n_threads  = 4;
    int draft_threads = 4;
    int spec_k = 4;
    float temperature = 0.0f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int repeat_last_n = 64;
    uint32_t seed = 42;
    bool bind_cores = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--draft-threads") == 0 && i + 1 < argc) {
            draft_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--spec-k") == 0 && i + 1 < argc) {
            spec_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            repeat_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
            repeat_last_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            input_tokens = parse_tokens(argv[++i]);
        } else if (strcmp(argv[i], "--bind-cores") == 0) {
            bind_cores = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
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

    // Native tokenizer not integrated — use --tokens
    if (input_tokens.empty()) {
        fprintf(stderr, "Warning: native tokenizer not yet integrated. "
                        "Use --tokens with pre-tokenized input.\n");
        fprintf(stderr, "Example: python3 -c \"\n"
                        "from transformers import AutoTokenizer\n"
                        "t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')\n"
                        "ids = t.apply_chat_template([{'role':'user','content':'%s'}], "
                        "tokenize=True, add_generation_prompt=True)\n"
                        "print(','.join(map(str, ids)))\"\n", prompt.c_str());
        return 1;
    }

    // Thread binding for reduced scheduling jitter
    if (bind_cores) {
#ifdef _OPENMP
        omp_set_num_threads(n_threads);
#endif
        setenv("OMP_PROC_BIND", "close", 0);
        setenv("OMP_PLACES", "cores", 0);
        setenv("GOMP_CPU_AFFINITY", "0-1023", 0);
        fprintf(stderr, "Thread binding: enabled (%d threads, close/cores)\n", n_threads);
    }

    // ── Speculative decoding path ───────────────────────────────
    if (!draft_path.empty()) {
        fprintf(stderr, "=== SPECULATIVE DECODING MODE ===\n");
        fprintf(stderr, "Loading target model: %s\n", model_path.c_str());
        diffuse_model * target_model = diffuse_model_load(model_path, n_threads);
        if (!target_model) {
            fprintf(stderr, "Failed to load target model\n");
            return 1;
        }

        fprintf(stderr, "Loading draft model: %s\n", draft_path.c_str());
        diffuse_model * draft_model = diffuse_model_load(draft_path, draft_threads);
        if (!draft_model) {
            fprintf(stderr, "Failed to load draft model\n");
            diffuse_model_free(target_model);
            return 1;
        }

        const auto & t_hp = diffuse_model_hparams(target_model);
        const auto & d_hp = diffuse_model_hparams(draft_model);
        int n_ctx = (int)input_tokens.size() + max_tokens + spec_k + 4;

        fprintf(stderr, "Target: %dL, %d heads, vocab=%d\n",
                (int)t_hp.n_layer, (int)t_hp.n_head, (int)t_hp.n_vocab);
        fprintf(stderr, "Draft:  %dL, %d heads, vocab=%d\n",
                (int)d_hp.n_layer, (int)d_hp.n_head, (int)d_hp.n_vocab);
        fprintf(stderr, "Prompt: %d tokens, max_generate: %d, K=%d\n",
                (int)input_tokens.size(), max_tokens, spec_k);

        diffuse_context * target_ctx = diffuse_context_new(target_model, n_ctx, n_threads);
        diffuse_context * draft_ctx  = diffuse_context_new(draft_model, n_ctx, draft_threads);

        ar_spec_params sparams;
        sparams.K = spec_k;

        ar_spec_stats stats;

        auto result = ar_speculative_generate(
            target_ctx, draft_ctx, input_tokens, max_tokens, sparams,
            [](int32_t token, int pos) -> bool {
                fprintf(stderr, "\r  generating... %d tokens", pos);
                return true;
            },
            &stats);

        fprintf(stderr, "\n\nGenerated %d token IDs:\n", (int)result.size());
        for (size_t i = 0; i < result.size(); i++) {
            if (i > 0) printf(",");
            printf("%d", result[i]);
        }
        printf("\n");

        fprintf(stderr, "\nSpeculative stats:\n");
        fprintf(stderr, "  Acceptance rate: %.1f%% (%d/%d)\n",
                100.0f * stats.acceptance_rate(),
                stats.total_accepted, stats.total_draft_tokens);
        fprintf(stderr, "  Tokens/target_call: %.2f\n",
                stats.tokens_per_target_call());
        fprintf(stderr, "  Fast rejects: %d\n", stats.total_fast_rejects);

        diffuse_context_free(target_ctx);
        diffuse_context_free(draft_ctx);
        diffuse_model_free(target_model);
        diffuse_model_free(draft_model);
        return 0;
    }

    // ── Standard AR decoding path ───────────────────────────────
    fprintf(stderr, "Loading model...\n");
    diffuse_model * model = diffuse_model_load(model_path, n_threads);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = diffuse_model_hparams(model);
    int n_ctx = (int)input_tokens.size() + max_tokens;
    fprintf(stderr, "Prompt: %d tokens, max_generate: %d, ctx: %d\n",
            (int)input_tokens.size(), max_tokens, n_ctx);

    diffuse_context * ctx = diffuse_context_new(model, n_ctx, n_threads);

    // Setup AR sampler params
    ar_sampler_params sparams;
    sparams.temperature    = temperature;
    sparams.top_p          = top_p;
    sparams.top_k          = top_k;
    sparams.repeat_penalty = repeat_penalty;
    sparams.repeat_last_n  = repeat_last_n;
    sparams.seed           = seed;

    // Generate with streaming output
    fprintf(stderr, "Generating (greedy=%s, temp=%.2f, top_p=%.2f, top_k=%d)...\n",
            temperature <= 0.0f ? "yes" : "no", temperature, top_p, top_k);

    auto result = ar_generate(ctx, input_tokens, max_tokens, sparams,
        [](int32_t token, int pos) -> bool {
            fprintf(stderr, "\r  generating... %d tokens", pos);
            return true;
        });

    fprintf(stderr, "\n\nGenerated %d token IDs:\n", (int)result.size());
    for (size_t i = 0; i < result.size(); i++) {
        if (i > 0) printf(",");
        printf("%d", result[i]);
    }
    printf("\n");

    fprintf(stderr, "\nDecode with:\n"
                    "  python3 -c \"from transformers import AutoTokenizer; "
                    "t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct'); "
                    "print(t.decode([...]))\"\n");

    // Cleanup
    diffuse_context_free(ctx);
    diffuse_model_free(model);

    return 0;
}
