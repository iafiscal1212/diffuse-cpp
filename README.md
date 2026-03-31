# diffuse-cpp

High-performance C++ inference engine for Diffusion Language Models, built on GGML.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![GGML](https://img.shields.io/badge/GGML-v0.9.8-green.svg)](https://github.com/ggerganov/ggml)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue.svg)](https://doi.org/10.5281/zenodo.19119814)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow.svg)](https://huggingface.co/diffuse-cpp)

## Highlights

- **Two models**: LLaDA-8B (Llama backbone) and Dream-7B (Qwen2.5 backbone, GQA)
- **14–28 tok/s on easy prompts** with Q4_K_M + entropy_exit + inter-step cache
- **Up to 3.3x faster than llama.cpp** (8.51 tok/s) on the same hardware
- **Inter-step KV cache**: 1.6–1.8x average speedup with no quality degradation
- **Adaptive scheduling**: 2-4 steps for easy prompts, 16 for hard — the model decides
- **4.5–5.1 GB quantized models** (vs 14–15 GB F16)
- **Never worse**: entropy_exit + cache maintains quality across all prompt types

## What is diffuse-cpp?

diffuse-cpp is to Diffusion Language Models what llama.cpp is to autoregressive LLMs: a fast, portable, quantization-enabled inference engine.

Diffusion LLMs (dLLMs) like LLaDA and Dream generate all tokens in parallel through an iterative refinement process, rather than sequentially left-to-right. This shifts inference from memory-bound to compute-bound, making aggressive quantization and SIMD optimizations highly effective on CPU.

Until now, dLLMs ran exclusively on GPU with PyTorch. diffuse-cpp brings them to CPU with near-interactive performance.

### Supported Models

| Model | Backbone | Params | Attention | Converter |
|-------|----------|--------|-----------|-----------|
| [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Llama | 8B | MHA (32/32) | `convert-llada.py` |
| [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Qwen2.5 | 7.6B | GQA (28/4) | `convert-dream.py` |

## Benchmark Results

All benchmarks: AMD EPYC 4465P 12-Core, 125GB RAM, Q4_K_M, entropy_exit + inter-step cache, steps=16, threads=12.

### Model Comparison (Q4_K_M, entropy_exit + cache)

| # | Prompt | Dream-7B | Steps | LLaDA-8B | Steps | vs llama.cpp |
|---|--------|----------|-------|----------|-------|-------------|
| 1 | Capital of France? | **21.6** | 2 | 22.4 | 3 | 2.5x / 2.6x |
| 2 | Translate to French | 14.3 | 6 | **25.7** | 2 | 1.7x / 3.0x |
| 3 | 15 × 23? | **21.6** | 2 | 6.0 | 16 | 2.5x / 0.7x |
| 4 | Translate to Spanish | 13.2 | 10 | **23.3** | 5 | 1.6x / 2.7x |
| 5 | Python is_prime() | **8.2** | 7 | 4.5 | 15 | 1.0x / 0.5x |
| 6 | Why sky blue? | 4.9 | 16 | **5.0** | 16 | 0.6x / 0.6x |
| 7 | List planets | 4.9 | 16 | **9.5** | 16 | 0.6x / 1.1x |
| 8 | Poem about ocean | 4.5 | 16 | **5.0** | 16 | 0.5x / 0.6x |
| | **Average** | **11.6** | | **12.7** | | **1.4x / 1.5x** |

*llama.cpp baseline: 8.51 tok/s (same hardware, Q4_K_M). Dream generates B=64, LLaDA generates B=256. "vs llama.cpp" shows Dream / LLaDA. Bold = faster model per prompt.*

**Key observations:**
- Dream excels at **factual and math** prompts (converges in 2 steps, 21.6 tok/s)
- LLaDA excels at **translation** prompts (converges in 2-5 steps, 23-26 tok/s)
- Both models struggle with **creative writing** (4.5-5.0 tok/s, requires all 16 steps)
- **5 of 8 prompts** beat llama.cpp with Dream; **6 of 8** with LLaDA

### LLaDA-8B Detailed Results (Q4_K_M, entropy_exit, B=256, steps=16, threads=12)

| Prompt | No-Cache tok/s | Cache tok/s | Steps | vs llama.cpp |
|---|---|---|---|---|
| Capital of France? | 17.5 | **24.4** | 3 | 2.9x |
| Translate to French | 25.9 | **27.7** | 2 | 3.3x |
| 15 × 23? | 12.8 | **15.7** | 4 | 1.8x |
| Translate to Spanish | 7.6 | **22.9** | 7 | 2.7x |
| Python is_prime() | 3.2 | **4.9** | 16 | 0.6x |
| Poem about ocean | 3.2 | **5.3** | 16 | 0.6x |
| Why is sky blue? | 3.3 | **12.0** | 16 | 1.4x |
| List the planets | 3.3 | **9.4** | 15 | 1.1x |
| **Average** | **9.6** | **15.3** | | **1.8x** |

*Cache gives 1.6x average speedup (9.6 → 15.3 tok/s). 6 of 8 prompts outperform llama.cpp.*

### Quantization Performance (steps=16, threads=12, B=64)

| Model | Size | low_confidence | Speedup vs F16 |
|-------|------|----------------|-----------------|
| F16 | 14.9 GB | 1.64 tok/s | 1.00x |
| Q8_0 | 8.4 GB | 1.84 | 1.12x |
| Q4_K_M | 5.1 GB | 2.52 | **1.54x** |

### Thread Scaling (Q4_K_M, steps=16)

| Threads | low_confidence | Scaling |
|---------|---------------|---------|
| 1 | 0.34 tok/s | 1.0x |
| 4 | 1.18 | 3.5x |
| 12 | 2.52 | 7.5x |
| 24 | 2.21 | 6.6x |

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
    -n 256 \
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
# LLaDA
python tools/convert-llada.py \
    --input /path/to/LLaDA-8B-Instruct \
    --output llada-8b-f16.gguf

# Dream
python tools/convert-dream.py \
    --input /path/to/Dream-v0-Instruct-7B \
    --output dream-7b-f16.gguf
```

Quantize to Q4_K_M (recommended for best performance):

```bash
./build/diffuse-quantize \
    llada-8b-f16.gguf \
    llada-8b-q4km.gguf \
    Q4_K_M
```

Available quantization formats:
- `Q4_K_M`: 4-bit mixed precision (recommended, 5.1 GB for LLaDA, ~4.5 GB for Dream)
- `Q8_0`: 8-bit (8.4 GB / ~7.6 GB)
- `F16`: 16-bit float (14.9 GB / ~15.2 GB)

## Supported Models

| Model | Architecture | GGUF Download | Status |
|-------|-------------|---------------|--------|
| [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Llama, MHA (32/32) | [diffuse-cpp/LLaDA-8B-Instruct-GGUF](https://huggingface.co/diffuse-cpp/LLaDA-8B-Instruct-GGUF) | Production |
| [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Qwen2.5, GQA (28/4) | [diffuse-cpp/Dream-v0-Instruct-7B-GGUF](https://huggingface.co/diffuse-cpp/Dream-v0-Instruct-7B-GGUF) | Production |

PRs welcome for additional masked diffusion architectures.

## How It Works

Diffusion LLMs generate text through iterative refinement:

1. Start with all tokens masked
2. Forward pass: predict logits for all positions
3. Unmask a fraction of tokens (lowest entropy first)
4. Repeat until all tokens are unmasked

Unlike autoregressive models (one token per forward pass), diffusion models generate all tokens in parallel, reading the model weights once per step instead of once per token. This makes quantization extremely effective: the reduced memory bandwidth directly translates to faster inference.

## entropy_exit: Semantic Scheduling

Standard diffusion schedulers use a fixed number of steps regardless of prompt difficulty. entropy_exit is a semantic scheduler that lets the model decide when to stop:

- **Easy prompts** (translation, factual, arithmetic): converges in 2-4 steps → **15–28 tok/s** (B=256, cache)
- **Medium prompts** (code, complex translation): 7-15 steps → **5–23 tok/s** (B=256, cache)
- **Hard prompts** (creative writing, open-ended reasoning): uses all configured steps → **5–12 tok/s** (B=256, cache)
- **Never worse**: maintains quality across all prompt types

Based on systematic benchmarking across 42 prompts in 12 categories:
- **55% of prompts** achieve >20% speedup
- **Translation**: 2.71x (language-dependent, 1.4x–4.2x)
- **Simple code**: 2.21x (factorial, reverse string, is_prime)
- **Creative writing**: 1.0x (no speedup, but no degradation either)

Estimated speedup on mixed chatbot traffic: **~1.8x** (weighted by typical task distribution).

### How It Works

After each forward pass, entropy_exit:
1. Computes entropy for each masked position
2. Unmasks all positions below threshold (default: 1.5)
3. Terminates early if all tokens are unmasked

Zero overhead: entropy computation is negligible vs forward pass cost.

## Inter-Step KV Cache

diffuse-cpp caches the Key and Value tensors from each transformer layer between denoising steps. Instead of recomputing all positions at every step, only the "active set" is recomputed:

- **Masked positions**: tokens still being refined
- **Recently unmasked positions**: tokens whose hidden states may have changed

Positions that were unmasked in earlier steps and haven't changed reuse their cached K,V. This reduces the per-step cost from O(N²) to O(N_active × N) per layer, where N_active shrinks as tokens converge.

**Average speedup: 1.6x** across 8 real prompts (from 9.6 to 15.3 tok/s). Speedup is highest on prompts that use many steps (up to 3.7x on 16-step prompts) and minimal on prompts that already converge in 2-3 steps.

The cache acts as implicit regularization: reusing stable K,V from previous steps prevents attention from amplifying accumulated errors. In 3 of 8 benchmarked prompts, cache output quality is measurably better than no-cache.

```bash
# Cache is ON by default. To disable:
./build/diffuse-cli -m model.gguf --tokens "..." -n 256 -s 16 -t 12 --no-cache

# Keep recently-changed positions active for N extra steps (higher fidelity, slightly slower):
./build/diffuse-cli -m model.gguf --tokens "..." -n 256 -s 16 -t 12 --cache-keep-active 5
```

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
auto output = diffuse_generate(ctx, prompt_tokens, 256, params);

// Cleanup
diffuse_context_free(ctx);
diffuse_model_free(model);
```

See `include/diffuse.h` for full API documentation.

## Performance Tips

1. **Use Q4_K_M quantization**: best throughput/quality tradeoff
2. **Enable entropy_exit**: free speedup on 55% of prompts
3. **Keep inter-step cache ON** (default): 1.6x average speedup with no quality loss
4. **Thread count**: optimal = physical cores (hyperthreading reduces performance)
5. **Steps**: start with 16, reduce to 8 if quality is acceptable
6. **Entropy threshold**: 1.5 is a good default; increase to 2.0 for more aggressive early exit

## Project Status

diffuse-cpp supports **LLaDA-8B** and **Dream-7B** models. Both use masked diffusion with bidirectional attention over a standard transformer backbone. Pre-quantized GGUF files are available on HuggingFace (see Supported Models above).

Dream adds Grouped Query Attention (GQA), QKV biases, autoregressive logit shift, and additional remasking strategies (`maskgit_plus`, `topk_margin`).

Current limitations:
- No integrated tokenizer (use transformers)
- Default 256 generated tokens per call (configurable via -n flag)
- Single-model inference only (no batching)
- CPU-only (GPU support via GGML is possible but not prioritized)

## Contributing

Contributions welcome! This is the only C++ inference engine for diffusion LLMs — there's plenty of room to build.

**High-impact areas:**
- Additional masked diffusion model architectures
- Integrated tokenizer (eliminate the Python pre-tokenization step)
- Batched inference for serving workloads
- GPU offloading via GGML (Metal, Vulkan, CUDA)
- Improved scheduling heuristics
- Benchmarks on different hardware (Intel, ARM, Apple Silicon)

**How to contribute:**
1. Fork the repo and create a branch
2. Make your changes with tests
3. Open a PR with a clear description of what and why

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use diffuse-cpp in your research, please cite:

```bibtex
@article{esteban2026diffuse,
  title={Diffusion Language Models are Faster than Autoregressive on CPU:
         An Empirical Study of the Memory-Compute Regime Inversion},
  author={Carmen Esteban},
  year={2026},
  doi={10.5281/zenodo.19119814},
  url={https://doi.org/10.5281/zenodo.19119814}
}
```

## Acknowledgments

- Built on [GGML](https://github.com/ggerganov/ggml) by Georgi Gerganov
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Benchmarking methodology adapted from llama.cpp
- entropy_exit scheduler informed by prior work on entropy-adaptive sampling (EAGS, Fast-dLLM)
