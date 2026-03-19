# diffuse-cpp

High-performance C++ inference engine for Diffusion Language Models, built on GGML.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![GGML](https://img.shields.io/badge/GGML-v0.9.8-green.svg)](https://github.com/ggerganov/ggml)

## Highlights

- **13.59 tok/s on CPU** with Q4_K_M quantization and entropy_exit scheduling
- **8.27x speedup** vs F16 baseline (1.64 tok/s)
- **5.1 GB quantized model** (vs 14.9 GB F16)
- **Semantic scheduling**: model decides when to stop (16→3 steps on deterministic prompts)
- **Zero training overhead**: no auxiliary models, no fine-tuning required
- **Never worse**: maintains quality across all prompt types

## What is diffuse-cpp?

diffuse-cpp is to Diffusion Language Models what llama.cpp is to autoregressive LLMs: a fast, portable, quantization-enabled inference engine.

Diffusion LLMs (dLLMs) like LLaDA generate all tokens in parallel through an iterative refinement process, rather than sequentially left-to-right. This shifts inference from memory-bound to compute-bound, making aggressive quantization and SIMD optimizations highly effective on CPU.

Until now, dLLMs ran exclusively on GPU with PyTorch. diffuse-cpp brings them to CPU with near-interactive performance.

## Benchmark Results

All benchmarks: LLaDA-8B-Instruct, 24-core Xeon, 125GB RAM, 64 tokens generated, 3 reps + 1 warmup.

### Table 1: Performance by Quantization (steps=16, threads=12)

| Model | Size | Scheduler | tok/s | Speedup vs F16 |
|-------|------|-----------|-------|-----------------|
| F16 | 14.9 GB | low_confidence | 1.64 | 1.00x |
| F16 | 14.9 GB | entropy_exit | 8.74 | 5.32x |
| Q8_0 | 8.4 GB | low_confidence | 1.86 | 1.13x |
| Q8_0 | 8.4 GB | entropy_exit | 10.09 | 6.14x |
| Q4_K_M | 5.1 GB | low_confidence | 2.48 | 1.51x |
| Q4_K_M | 5.1 GB | entropy_exit | 13.59 | 8.27x |

### Table 2: Thread Scaling (Q4_K_M, steps=16)

| Threads | low_confidence | entropy_exit | Scaling |
|---------|---------------|--------------|---------|
| 1 | 0.34 tok/s | 1.78 tok/s | 1.0x |
| 4 | 1.17 | 6.35 | 3.6x |
| 12 | 2.48 | 13.59 | 7.6x |
| 24 | 2.05 | 11.38 | 6.4x |

### Table 3: Steps vs Throughput (Q4_K_M, threads=12)

| Steps | low_confidence | entropy_exit | Actual Steps |
|-------|---------------|--------------|-------------|
| 8 | 5.09 tok/s | 12.75 tok/s | 3 |
| 16 | 2.48 | 13.59 | 3 |
| 32 | 1.24 | 12.76 | 3 |

## Quick Start

### Build

```bash
git clone --recursive https://github.com/iafiscal1212/diffuse-cpp.git
cd diffuse-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Basic Usage

```bash
# Pre-tokenize your prompt (diffuse-cpp does not include a tokenizer)
# Use transformers or tiktoken to get token IDs as a comma-separated list

# Generate text (example with token IDs)
./build/diffuse-cli \
    -m model.gguf \
    --tokens "128000,3923,374,279,6864,315,9822,30" \
    -n 64 \
    -s 16 \
    -t 12 \
    --remasking entropy_exit
```

**Note**: diffuse-cpp operates on token IDs, not raw text. Use the HuggingFace transformers library to tokenize your prompts:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("LLaDA-8B-Instruct")
tokens = tokenizer.encode("What is the capital of France?")
print(",".join(map(str, tokens)))
```

## Quantization

Convert a model from HuggingFace to GGUF F16:

```bash
python tools/convert-llada.py \
    --input /path/to/LLaDA-8B-Instruct \
    --output llada-8b-f16.gguf
```

Quantize to Q4_K_M (recommended for best performance):

```bash
./build/diffuse-quantize \
    llada-8b-f16.gguf \
    llada-8b-q4km.gguf \
    Q4_K_M
```

Available quantization formats:
- `Q4_K_M`: 4-bit mixed precision (recommended, 5.1 GB)
- `Q8_0`: 8-bit (8.4 GB, higher quality)
- `F16`: 16-bit float (14.9 GB, reference quality)

## Supported Models

| Model | Architecture | Status |
|-------|-------------|--------|
| LLaDA-8B | Masked diffusion (Llama backbone) | Production |
| SEDD | Score-based diffusion | Planned |
| MDLM | Masked discrete diffusion | Planned |

## How It Works

Diffusion LLMs generate text through iterative refinement:

1. Start with all tokens masked
2. Forward pass: predict logits for all positions
3. Unmask a fraction of tokens (lowest entropy first)
4. Repeat until all tokens are unmasked

Unlike autoregressive models (one token per forward pass), diffusion models generate all tokens in parallel, reading the model weights once per step instead of once per token. This makes quantization extremely effective: the reduced memory bandwidth directly translates to faster inference.

## entropy_exit: Semantic Scheduling

Standard diffusion schedulers use a fixed number of steps regardless of prompt difficulty. entropy_exit is a semantic scheduler that lets the model decide when to stop:

- **Deterministic prompts** (translation, factual questions): converges in 2-3 steps
- **Creative prompts** (story writing, open-ended reasoning): uses all configured steps
- **Never worse**: maintains quality across all prompt types

Based on systematic benchmarking across 42 prompts in 12 categories:
- **55% of prompts** achieve >20% speedup
- **Instruction following**: 3.86x average speedup (passive voice, question transformation)
- **Classification**: 2.84x (sentiment, spam detection, NER)
- **Translation**: 2.71x (language-dependent, 1.4x–4.2x)
- **Simple code**: 2.21x (factorial, reverse string, is_prime)
- **Creative writing**: 1.0x (no speedup, but no degradation either)

Estimated speedup on real chatbot traffic: **1.8x** (weighted by typical task distribution).

### How It Works

After each forward pass, entropy_exit:
1. Computes entropy for each masked position
2. Unmasks all positions below threshold (default: 1.5)
3. Terminates early if all tokens are unmasked

Zero overhead: entropy computation is negligible vs forward pass cost.

## Building From Source

### Requirements

- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.14+
- Git (for GGML submodule)

### Full Build

```bash
git clone --recursive https://github.com/iafiscal1212/diffuse-cpp.git
cd diffuse-cpp

# Release build (optimized)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Tools (CLI, bench, quantize) are built by default
# To disable: -DDIFFUSE_BUILD_TOOLS=OFF
```

Binaries:
- `build/diffuse-cli`: command-line inference
- `build/diffuse-quantize`: model quantization
- `build/diffuse-bench`: benchmarking

## API

diffuse-cpp provides a C++ API for embedding in other applications:

```cpp
#include "diffuse.h"

// Load model
diffuse_model* model = diffuse_model_load("model.gguf", 12);

// Create context
diffuse_context* ctx = diffuse_context_new(model, 128, 12);

// Configure sampler
diffuse_sampler_params params;
params.n_steps = 16;
params.remasking = diffuse_remasking::ENTROPY_EXIT;
params.entropy_threshold = 1.5f;

// Generate
std::vector<int32_t> prompt_tokens = {128000, 3923, 374};
auto output = diffuse_generate(ctx, prompt_tokens, 64, params);

// Cleanup
diffuse_context_free(ctx);
diffuse_model_free(model);
```

See `include/diffuse.h` for full API documentation.

## Performance Tips

1. **Use Q4_K_M quantization**: best throughput/quality tradeoff
2. **Enable entropy_exit**: free speedup on 55% of prompts
3. **Thread count**: optimal = physical cores (hyperthreading reduces performance)
4. **Steps**: start with 16, reduce to 8 if quality is acceptable
5. **Entropy threshold**: 1.5 is a good default; increase to 2.0 for more aggressive early exit

## Project Status

diffuse-cpp is production-ready for LLaDA-8B models. Additional architectures (SEDD, MDLM) are planned.

Current limitations:
- No integrated tokenizer (use transformers)
- Single-model inference only (no batching)
- CPU-only (GPU support via GGML is possible but not prioritized)

## Contributing

Contributions welcome. Areas of interest:
- Additional model architectures (SEDD, MDLM)
- Improved scheduling heuristics
- Tokenizer integration
- Multi-query batching

## License

AGPL-3.0. Commercial licenses available upon request.

For commercial licensing inquiries, contact: caresment@gmail.com

## Citation

If you use diffuse-cpp in your research, please cite:

```bibtex
@software{diffuse_cpp_2026,
  title={diffuse-cpp: High-Performance Inference for Diffusion Language Models},
  author={Carmen Estévez},
  year={2026},
  url={https://github.com/iafiscal1212/diffuse-cpp}
}
```

## Acknowledgments

- Built on [GGML](https://github.com/ggerganov/ggml) by Georgi Gerganov
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Benchmarking methodology adapted from llama.cpp
- entropy_exit scheduler informed by prior work on entropy-adaptive sampling (EAGS, Fast-dLLM)
