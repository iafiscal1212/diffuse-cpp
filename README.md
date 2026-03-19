# diffuse-cpp

High-performance C++ inference engine for Diffusion Language Models, built on GGML.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![GGML](https://img.shields.io/badge/GGML-v0.9.8-green.svg)](https://github.com/ggerganov/ggml)

## Highlights

- **~10 tok/s on real prompts** with Q4_K_M quantization and entropy_exit scheduling
- **~6x speedup** vs F16 baseline (1.64 tok/s) on easy prompts (factual, translation, arithmetic)
- **Adaptive scheduling**: 3-4 steps for easy prompts, 16 for hard — the model decides
- **5.1 GB quantized model** (vs 14.9 GB F16)
- **Never worse**: entropy_exit maintains quality across all prompt types
- **40+ tok/s peak** on synthetic benchmarks (single forward pass)

## What is diffuse-cpp?

diffuse-cpp is to Diffusion Language Models what llama.cpp is to autoregressive LLMs: a fast, portable, quantization-enabled inference engine.

Diffusion LLMs (dLLMs) like LLaDA generate all tokens in parallel through an iterative refinement process, rather than sequentially left-to-right. This shifts inference from memory-bound to compute-bound, making aggressive quantization and SIMD optimizations highly effective on CPU.

Until now, dLLMs ran exclusively on GPU with PyTorch. diffuse-cpp brings them to CPU with near-interactive performance.

## Benchmark Results

All benchmarks: LLaDA-8B-Instruct, AMD EPYC 4465P 12-Core (24 threads), 125GB RAM, 64 tokens generated, 3 reps + 1 warmup.

### Quantization Performance (steps=16, threads=12)

| Model | Size | low_confidence | entropy_exit (real prompts) | Speedup vs F16 |
|-------|------|----------------|----------------------------|-----------------|
| F16 | 14.9 GB | 1.64 tok/s | — | 1.00x |
| Q8_0 | 8.4 GB | 1.84 | — | 1.12x |
| Q4_K_M | 5.1 GB | 2.52 | 9-11 tok/s (easy) | **~6x** |

### Real Prompt Performance (Q4_K_M, steps=16, threads=12)

| Prompt type | entropy_exit tok/s | Steps used | Speedup vs baseline |
|---|---|---|---|
| Factual ("Capital of France?") | **9.22** | 4 | 3.9x |
| Translation ("Translate to French") | **10.23** | 3 | 4.6x |
| Arithmetic ("15 × 23?") | **11.49** | 3 | 5.5x |
| Code (is_prime function) | **2.53** | 15 | 1.1x |
| Creative (poem, explanation) | 2.33 | 17 | 1.0x |

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

- **Easy prompts** (translation, factual, arithmetic): converges in 3-4 steps → **9-11 tok/s**
- **Medium prompts** (code, complex translation): 8-15 steps → **2.5-4.6 tok/s**
- **Hard prompts** (creative writing, open-ended reasoning): uses all configured steps → **~2.3 tok/s**
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
- Maximum 128 generated tokens per call (GGML buffer limitation, fixable)
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
