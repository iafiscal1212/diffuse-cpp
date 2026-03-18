#include "diffuse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>

static std::vector<int> parse_list(const char * str) {
    std::vector<int> vals;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        vals.push_back(atoi(item.c_str()));
    }
    return vals;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::vector<int> steps_list = {8, 16, 32};
    std::vector<int> threads_list = {1, 4};
    int n_generate = 128;
    int n_repeat   = 3;
    int prompt_len = 32;  // dummy prompt length

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            steps_list = parse_list(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads_list = parse_list(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_generate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            n_repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) {
            prompt_len = atoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: diffuse-bench -m MODEL.gguf [-s steps] [-t threads] [-n tokens] [-r repeats]\n");
        return 1;
    }

    // Load model with max threads
    int max_threads = *std::max_element(threads_list.begin(), threads_list.end());
    diffuse_model * model = diffuse_model_load(model_path, max_threads);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = diffuse_model_hparams(model);

    // Dummy prompt (token ID 1 repeated)
    std::vector<int32_t> prompt(prompt_len, 1);

    // Print header
    printf("%-8s %-8s %-8s %-12s %-12s %-12s\n",
           "steps", "threads", "gen_tok", "time_ms", "tok/s", "step_ms");
    printf("%-8s %-8s %-8s %-12s %-12s %-12s\n",
           "-----", "-------", "-------", "-------", "-----", "-------");

    for (int n_steps : steps_list) {
        for (int n_threads : threads_list) {
            double total_ms = 0.0;

            for (int rep = 0; rep < n_repeat; rep++) {
                int n_ctx = prompt_len + n_generate;
                diffuse_context * ctx = diffuse_context_new(model, n_ctx, n_threads);

                diffuse_sampler_params sparams;
                sparams.n_steps = n_steps;
                sparams.seed    = 42 + rep;

                auto t0 = std::chrono::high_resolution_clock::now();

                auto result = diffuse_generate(ctx, prompt, n_generate, sparams);

                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;

                diffuse_context_free(ctx);
            }

            double avg_ms     = total_ms / n_repeat;
            double tok_per_s  = n_generate / (avg_ms / 1000.0);
            double step_ms    = avg_ms / n_steps;

            printf("%-8d %-8d %-8d %-12.1f %-12.2f %-12.1f\n",
                   n_steps, n_threads, n_generate, avg_ms, tok_per_s, step_ms);
        }
    }

    diffuse_model_free(model);
    return 0;
}
