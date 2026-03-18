# diffuse-cpp

High-performance C++ inference engine for Diffusion Language Models.

Supports LLaDA, SEDD, and MDLM architectures using GGML for quantized tensor operations.

## Why diffuse-cpp?

Diffusion LLMs read weights once per step and generate all tokens in parallel, shifting inference from memory-bound to compute-bound. With quantization (Q4_K_M) and SIMD, this yields **3x faster inference** than autoregressive engines on CPU.

## Build

```bash
git clone --recursive https://github.com/iafiscal1212/diffuse-cpp.git
cd diffuse-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usage

### Convert model
```bash
python tools/convert-llada.py \
    --input /path/to/LLaDA-8B-Instruct \
    --output llada-8b-f32.gguf
```

### Generate text
```bash
./build/diffuse-cli -m llada-8b-f32.gguf -p "Explain quantum computing" -s 32 -t 12
```

### Benchmark
```bash
./build/diffuse-bench -m llada-8b-f32.gguf -s 8,16,32 -t 1,4,12 -r 3
```

## Supported models

| Model | Architecture | Status |
|-------|-------------|--------|
| LLaDA-8B | Masked diffusion (Llama backbone) | Phase 1 |
| SEDD | Score-based diffusion | Phase 3 |
| MDLM | Masked discrete diffusion | Phase 3 |

## License

MIT
