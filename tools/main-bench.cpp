// diffuse-bench: Exhaustive benchmark for diffuse-cpp
//
// Measures tokens/s across steps, threads, and schedulers.
// Outputs markdown table and JSON for publication-ready results.
//
// Usage: diffuse-bench -m model.gguf [options]

#include "diffuse.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

static std::vector<int> parse_int_list(const char * str) {
    std::vector<int> vals;
    const char * p = str;
    while (*p) {
        vals.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return vals;
}

struct bench_result {
    int n_steps;
    int n_threads;
    std::string scheduler;
    double elapsed_ms;
    double tok_per_sec;
    int actual_steps;
};

int main(int argc, char ** argv) {
    std::string model_path;
    std::vector<int32_t> input_tokens;
    int n_generate = 64;
    std::vector<int> steps_list = {8, 16, 32};
    std::vector<int> threads_list = {1, 4, 12};
    int n_reps = 3;
    int n_warmup = 1;
    bool json_output = false;
    float entropy_threshold = 1.5f;
    int prompt_len = 32;  // for dummy prompt if --tokens not given

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            input_tokens = parse_tokens(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_generate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            steps_list = parse_int_list(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads_list = parse_int_list(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            n_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            n_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--entropy-threshold") == 0 && i + 1 < argc) {
            entropy_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) {
            prompt_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s -m MODEL [options]\n", argv[0]);
            fprintf(stderr, "  --tokens IDs   Pre-tokenized input (comma-separated)\n");
            fprintf(stderr, "  --prompt-len N Dummy prompt length if no --tokens (default: 32)\n");
            fprintf(stderr, "  -n INT         Tokens to generate (default: 64)\n");
            fprintf(stderr, "  -s LIST        Step counts (default: 8,16,32)\n");
            fprintf(stderr, "  -t LIST        Thread counts (default: 1,4,12)\n");
            fprintf(stderr, "  -r INT         Repetitions (default: 3)\n");
            fprintf(stderr, "  --warmup INT   Warmup runs (default: 1)\n");
            fprintf(stderr, "  --json         Output JSON instead of table\n");
            fprintf(stderr, "  --entropy-threshold F  (default: 1.5)\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: -m MODEL required\n");
        return 1;
    }

    // If no tokens given, use dummy prompt
    if (input_tokens.empty()) {
        input_tokens.resize(prompt_len, 1);
        fprintf(stderr, "No --tokens given, using dummy prompt (%d tokens)\n", prompt_len);
    }

    // Schedulers to benchmark
    struct sched_config {
        const char * name;
        diffuse_remasking remasking;
    };
    sched_config schedulers[] = {
        {"low_confidence", diffuse_remasking::LOW_CONFIDENCE},
        {"entropy_exit",   diffuse_remasking::ENTROPY_EXIT},
    };
    int n_schedulers = 2;

    // Load model
    int max_threads = *std::max_element(threads_list.begin(), threads_list.end());
    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    diffuse_model * model = diffuse_model_load(model_path, max_threads);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = diffuse_model_hparams(model);
    int n_ctx = (int)input_tokens.size() + n_generate;

    fprintf(stderr, "  n_vocab=%d n_embd=%d n_layer=%d\n", hp.n_vocab, hp.n_embd, hp.n_layer);
    fprintf(stderr, "  prompt=%d gen=%d\n", (int)input_tokens.size(), n_generate);
    fprintf(stderr, "  steps=%zu configs, threads=%zu configs, reps=%d, warmup=%d\n",
            steps_list.size(), threads_list.size(), n_reps, n_warmup);
    fprintf(stderr, "\n");

    std::vector<bench_result> results;

    int total_configs = n_schedulers * (int)steps_list.size() * (int)threads_list.size();
    int config_idx = 0;

    for (int si = 0; si < n_schedulers; si++) {
        for (int n_steps : steps_list) {
            for (int n_threads : threads_list) {
                config_idx++;
                fprintf(stderr, "[%d/%d] %s s=%d t=%d ...",
                        config_idx, total_configs,
                        schedulers[si].name, n_steps, n_threads);
                fflush(stderr);

                diffuse_context * ctx = diffuse_context_new(model, n_ctx, n_threads);

                diffuse_sampler_params sparams;
                sparams.n_steps = n_steps;
                sparams.temperature = 0.0f;
                sparams.seed = 42;
                sparams.schedule = diffuse_schedule::COSINE;
                sparams.remasking = schedulers[si].remasking;
                sparams.entropy_threshold = entropy_threshold;

                // Warmup
                for (int w = 0; w < n_warmup; w++) {
                    diffuse_generate(ctx, input_tokens, n_generate, sparams);
                }

                // Timed runs
                double total_ms = 0;
                int last_actual_steps = 0;

                for (int r = 0; r < n_reps; r++) {
                    int actual_steps = 0;

                    auto t0 = std::chrono::high_resolution_clock::now();
                    diffuse_generate(ctx, input_tokens, n_generate, sparams,
                        [&actual_steps](int step, int, const std::vector<int32_t> &) {
                            actual_steps = step;
                        });
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    total_ms += ms;
                    last_actual_steps = actual_steps;

                    bench_result res;
                    res.n_steps = n_steps;
                    res.n_threads = n_threads;
                    res.scheduler = schedulers[si].name;
                    res.elapsed_ms = ms;
                    res.tok_per_sec = n_generate / (ms / 1000.0);
                    res.actual_steps = actual_steps;
                    results.push_back(res);
                }

                double avg_ms = total_ms / n_reps;
                double avg_tps = n_generate / (avg_ms / 1000.0);
                fprintf(stderr, " %.0fms (%.2f tok/s, %d steps)\n",
                        avg_ms, avg_tps, last_actual_steps);

                diffuse_context_free(ctx);
            }
        }
    }

    // Output
    if (json_output) {
        printf("{\n");
        printf("  \"model\": \"%s\",\n", model_path.c_str());
        printf("  \"n_prompt\": %d,\n", (int)input_tokens.size());
        printf("  \"n_generate\": %d,\n", n_generate);
        printf("  \"n_reps\": %d,\n", n_reps);
        printf("  \"results\": [\n");
        for (size_t i = 0; i < results.size(); i++) {
            const auto & r = results[i];
            printf("    {\"scheduler\": \"%s\", \"steps\": %d, \"threads\": %d, "
                   "\"elapsed_ms\": %.1f, \"tok_per_sec\": %.2f, \"actual_steps\": %d}%s\n",
                   r.scheduler.c_str(), r.n_steps, r.n_threads,
                   r.elapsed_ms, r.tok_per_sec, r.actual_steps,
                   i + 1 < results.size() ? "," : "");
        }
        printf("  ]\n}\n");
    } else {
        // Markdown summary table (averages over reps)
        printf("\n## Benchmark Results\n\n");
        printf("| Scheduler | Steps | Threads | Avg Time (ms) | tok/s | Actual Steps |\n");
        printf("|---|---|---|---|---|---|\n");

        for (int si = 0; si < n_schedulers; si++) {
            for (int n_steps : steps_list) {
                for (int n_threads : threads_list) {
                    double sum_ms = 0;
                    int count = 0;
                    int actual = 0;
                    for (const auto & r : results) {
                        if (r.scheduler == schedulers[si].name &&
                            r.n_steps == n_steps && r.n_threads == n_threads) {
                            sum_ms += r.elapsed_ms;
                            actual = r.actual_steps;
                            count++;
                        }
                    }
                    if (count > 0) {
                        double avg_ms = sum_ms / count;
                        double avg_tps = n_generate / (avg_ms / 1000.0);
                        printf("| %s | %d | %d | %.0f | %.2f | %d |\n",
                               schedulers[si].name, n_steps, n_threads,
                               avg_ms, avg_tps, actual);
                    }
                }
            }
        }
    }

    diffuse_model_free(model);
    return 0;
}
