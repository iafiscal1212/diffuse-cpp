# Benchmark Results — diffuse-cpp v0.2.0

**System**: AMD EPYC 4465P 12-Core (24 threads), 125GB RAM, Ubuntu 24.04, GCC 13.3
**Model**: LLaDA-8B-Instruct, 64/256 tokens generated, 32-token prompt
**Protocol**: 3 reps + 1 warmup per config, exclusive CPU (no competing processes)
**Date**: 2026-03-22, commit 054a80a+ (inter-step cache)

---

## Table 1: Synthetic Benchmark — Performance by Quantization (steps=16, threads=12)

> **Note**: Synthetic benchmark uses a dummy prompt (all tokens=1). entropy_exit converges
> in 1 step on this trivial input. See Table 5 for real-prompt performance.

| Model | Size (GB) | Scheduler | Time (ms) | tok/s | Actual Steps | Speedup vs F16 |
|---|---|---|---|---|---|---|
| F16 | 14.9 | low_confidence | 38935 | 1.64 | 16 | 1.00x |
| F16 | 14.9 | entropy_exit | 2594 | 24.94 | 1 | 15.21x |
| Q8_0 | 8.4 | low_confidence | 34910 | 1.84 | 16 | 1.12x |
| Q8_0 | 8.4 | entropy_exit | 2120 | 30.19 | 1 | 18.41x |
| Q4_K_M | 5.1 | low_confidence | 25374 | 2.52 | 16 | 1.54x |
| Q4_K_M | 5.1 | entropy_exit | 1855 | 35.82 | 1 | 21.84x |

## Table 2: Thread Scaling (Q4_K_M, steps=16)

| Threads | low_confidence | entropy_exit | Scaling (LC) |
|---|---|---|---|
| 1 | 0.34 tok/s | 5.26 tok/s | 1.0x |
| 4 | 1.18 | 19.05 | 3.5x |
| 12 | 2.52 | 35.82 | 7.5x |
| 24 | 2.21 | 32.92 | 6.6x |

Optimal threads = physical cores (12). Hyperthreading (24) degrades performance.

## Table 3: Diffusion Steps vs Throughput (Q4_K_M, threads=12)

| Configured Steps | low_confidence | entropy_exit | Actual Steps (EE) |
|---|---|---|---|
| 8 | 4.95 tok/s | 39.90 tok/s | 1 |
| 16 | 2.52 | 35.82 | 1 |
| 32 | 1.26 | 40.64 | 1 |

entropy_exit converges to 1 step on synthetic prompt regardless of configured steps.
Configured steps act as an upper bound — real prompts use 3-17 steps (see Table 5).

## Table 4: Per-Step Forward Pass Time (threads=12)

This is the fundamental metric — time for one transformer forward pass.

| Model | Size (GB) | Per-step (ms) | Derived from |
|---|---|---|---|
| F16 | 14.9 | 2433 | 38935ms / 16 steps |
| Q8_0 | 8.4 | 2182 | 34910ms / 16 steps |
| Q4_K_M | 5.1 | 1586 | 25374ms / 16 steps |

Throughput at N steps = 64 tokens / (N × per-step time).

## Table 5: Real Prompt Performance (Q4_K_M, steps=16, threads=12)

End-to-end via generate.py (includes tokenizer load + process startup, ~500-1500ms overhead).

| Prompt | low_confidence | entropy_exit | Steps (EE) | Speedup | Quality |
|---|---|---|---|---|---|
| "Capital of France?" | 2.38 tok/s | **9.22** | 4 | **3.9x** | Correct both |
| "Translate to French" | 2.20 | **10.23** | 3 | **4.6x** | Correct both |
| "Python is_prime()" | 2.25 | **2.53** | 15 | 1.1x | EE correct, LC empty |
| "Short poem about ocean" | 2.34 | 2.33 | 17 | 1.0x | Both mediocre |
| "Why is the sky blue?" | 2.12 | 2.21 | 17 | 1.0x | Both have artifacts |
| "List the planets" | 2.33 | 2.33 | 17 | 1.0x | LC better |
| "15 × 23 = ?" | 2.08 | **11.49** | 3 | **5.5x** | Correct both |
| "Translate to Spanish" | 2.32 | **4.59** | 8 | **2.0x** | Both mediocre |

**Pattern**: Easy prompts (factual, arithmetic, simple translation) converge in 3-4 steps.
Hard prompts (creative, explanatory, long-form) use all 16-17 steps.

### Real-Prompt Speedup vs F16 Baseline

F16 baseline (s=16, t=12, low_confidence) = 1.64 tok/s

| Scenario | Typical Steps | Q4_K_M EE tok/s | vs F16 baseline |
|---|---|---|---|
| Easy (factual, math, translation) | 3-4 | 9-11 | **~6x** |
| Medium (code, complex translation) | 8-15 | 2.5-4.6 | **~2x** |
| Hard (creative, explanatory) | 16-17 | 2.3 | **1.4x** |

## Table 6: Full Benchmark Matrix (Synthetic)

| Model | Steps | Threads | Scheduler | Time (ms) | tok/s |
|---|---|---|---|---|---|
| F16 | 8 | 1 | low_confidence | 126110 | 0.51 |
| F16 | 8 | 4 | low_confidence | 36303 | 1.76 |
| F16 | 8 | 12 | low_confidence | 19555 | 3.27 |
| F16 | 8 | 24 | low_confidence | 22821 | 2.80 |
| F16 | 16 | 1 | low_confidence | 251476 | 0.25 |
| F16 | 16 | 4 | low_confidence | 73391 | 0.87 |
| F16 | 16 | 12 | low_confidence | 38935 | 1.64 |
| F16 | 16 | 24 | low_confidence | 43862 | 1.46 |
| F16 | 32 | 1 | low_confidence | 499061 | 0.13 |
| F16 | 32 | 4 | low_confidence | 146057 | 0.44 |
| F16 | 32 | 12 | low_confidence | 76697 | 0.83 |
| F16 | 32 | 24 | low_confidence | 87184 | 0.73 |
| F16 | 8 | 1 | entropy_exit | 16336 | 3.92 |
| F16 | 8 | 4 | entropy_exit | 4575 | 13.99 |
| F16 | 8 | 12 | entropy_exit | 2639 | 24.25 |
| F16 | 8 | 24 | entropy_exit | 2708 | 23.64 |
| F16 | 16 | 1 | entropy_exit | 15972 | 4.01 |
| F16 | 16 | 4 | entropy_exit | 4464 | 14.34 |
| F16 | 16 | 12 | entropy_exit | 2594 | 24.94 |
| F16 | 16 | 24 | entropy_exit | 2692 | 23.77 |
| F16 | 32 | 1 | entropy_exit | 15921 | 4.02 |
| F16 | 32 | 4 | entropy_exit | 4454 | 14.37 |
| F16 | 32 | 12 | entropy_exit | 2386 | 26.82 |
| F16 | 32 | 24 | entropy_exit | 2527 | 25.32 |
| Q8_0 | 8 | 1 | low_confidence | 135207 | 0.47 |
| Q8_0 | 8 | 4 | low_confidence | 37747 | 1.70 |
| Q8_0 | 8 | 12 | low_confidence | 16974 | 3.77 |
| Q8_0 | 8 | 24 | low_confidence | 18826 | 3.40 |
| Q8_0 | 16 | 1 | low_confidence | 269926 | 0.24 |
| Q8_0 | 16 | 4 | low_confidence | 75543 | 0.85 |
| Q8_0 | 16 | 12 | low_confidence | 34910 | 1.84 |
| Q8_0 | 16 | 24 | low_confidence | 34552 | 1.85 |
| Q8_0 | 32 | 1 | low_confidence | 539810 | 0.12 |
| Q8_0 | 32 | 4 | low_confidence | 151846 | 0.42 |
| Q8_0 | 32 | 12 | low_confidence | 69624 | 0.92 |
| Q8_0 | 32 | 24 | low_confidence | 72621 | 0.88 |
| Q8_0 | 8 | 1 | entropy_exit | 17348 | 3.69 |
| Q8_0 | 8 | 4 | entropy_exit | 4714 | 13.58 |
| Q8_0 | 8 | 12 | entropy_exit | 2127 | 30.09 |
| Q8_0 | 8 | 24 | entropy_exit | 2174 | 29.46 |
| Q8_0 | 16 | 1 | entropy_exit | 17172 | 3.73 |
| Q8_0 | 16 | 4 | entropy_exit | 4740 | 13.51 |
| Q8_0 | 16 | 12 | entropy_exit | 2120 | 30.19 |
| Q8_0 | 16 | 24 | entropy_exit | 2160 | 29.63 |
| Q8_0 | 32 | 1 | entropy_exit | 17117 | 3.74 |
| Q8_0 | 32 | 4 | entropy_exit | 4665 | 13.72 |
| Q8_0 | 32 | 12 | entropy_exit | 2119 | 30.20 |
| Q8_0 | 32 | 24 | entropy_exit | 2597 | 25.98 |
| Q4_K_M | 8 | 1 | low_confidence | 95567 | 0.67 |
| Q4_K_M | 8 | 4 | low_confidence | 26911 | 2.38 |
| Q4_K_M | 8 | 12 | low_confidence | 12924 | 4.95 |
| Q4_K_M | 8 | 24 | low_confidence | 15991 | 4.02 |
| Q4_K_M | 16 | 1 | low_confidence | 190626 | 0.34 |
| Q4_K_M | 16 | 4 | low_confidence | 54129 | 1.18 |
| Q4_K_M | 16 | 12 | low_confidence | 25374 | 2.52 |
| Q4_K_M | 16 | 24 | low_confidence | 28908 | 2.21 |
| Q4_K_M | 32 | 1 | low_confidence | 378887 | 0.17 |
| Q4_K_M | 32 | 4 | low_confidence | 109403 | 0.58 |
| Q4_K_M | 32 | 12 | low_confidence | 50944 | 1.26 |
| Q4_K_M | 32 | 24 | low_confidence | 62587 | 1.02 |
| Q4_K_M | 8 | 1 | entropy_exit | 12499 | 5.12 |
| Q4_K_M | 8 | 4 | entropy_exit | 3411 | 18.76 |
| Q4_K_M | 8 | 12 | entropy_exit | 1605 | 39.90 |
| Q4_K_M | 8 | 24 | entropy_exit | 1990 | 32.19 |
| Q4_K_M | 16 | 1 | entropy_exit | 12171 | 5.26 |
| Q4_K_M | 16 | 4 | entropy_exit | 3360 | 19.05 |
| Q4_K_M | 16 | 12 | entropy_exit | 1855 | 35.82 |
| Q4_K_M | 16 | 24 | entropy_exit | 1944 | 32.92 |
| Q4_K_M | 32 | 1 | entropy_exit | 12092 | 5.29 |
| Q4_K_M | 32 | 4 | entropy_exit | 3415 | 18.76 |
| Q4_K_M | 32 | 12 | entropy_exit | 1575 | 40.64 |
| Q4_K_M | 32 | 24 | entropy_exit | 1817 | 35.22 |

## Table 7: Inter-Step Cache Performance (v0.2.0)

Q4_K_M, entropy_exit, n=256, steps=16, threads=12, seed=42. Cache caches K,V tensors between denoising steps and only recomputes active positions (masked + recently changed).

| # | Prompt | No-Cache tok/s | Cache tok/s | Speedup | Steps |
|---|--------|----------------|-------------|---------|-------|
| 1 | Capital of France? | 17.5 | **24.4** | 1.39x | 3 |
| 2 | Translate to French | 25.9 | **27.7** | 1.07x | 2 |
| 3 | Python is_prime() | 3.2 | **4.9** | 1.52x | 16 |
| 4 | Poem about ocean | 3.2 | **5.3** | 1.63x | 16 |
| 5 | Why is sky blue? | 3.3 | **12.0** | 3.66x | 16 |
| 6 | List the planets | 3.3 | **9.4** | 2.84x | 15 |
| 7 | 15 × 23? | 12.8 | **15.7** | 1.22x | 4 |
| 8 | Translate to Spanish | 7.6 | **22.9** | 3.03x | 7 |
| | **Average** | **9.6** | **15.3** | **1.59x** | |

### Analysis

- **More steps = more cache benefit**: 2-step prompts get 1.07x, 16-step prompts get up to 3.66x
- **Regularization effect**: cache improves output quality in 3 of 8 prompts (sky blue, planets, Spanish translation). Reusing stable K,V prevents attention from amplifying accumulated errors
- **vs llama.cpp** (8.51 tok/s): 6 of 8 prompts outperform with cache (up to 3.3x). 2 prompts (code generation, creative writing) remain slower due to requiring all 16 steps
- **Tuning**: `--cache-keep-active 5` increases token fidelity to 59.4% vs baseline at 1.64x speedup. Default (keep=0) maximizes speed at 1.59x

## Summary

### Synthetic Benchmark (dummy prompt)
- **Peak throughput**: 40.64 tok/s (Q4_K_M s=32 t=12 entropy_exit, 1 step)
- **F16 baseline**: 1.64 tok/s (s=16, t=12, low_confidence)
- **Thread scaling**: 7.5x from t=1 to t=12 (Q4_K_M)

### Real Prompts (end-to-end, B=256)
- **Easy prompts (cache)**: 15–28 tok/s, 2-4 steps, **up to 3.3x vs llama.cpp**
- **Medium prompts (cache)**: 5–23 tok/s, 7-15 steps
- **Hard prompts (cache)**: 5–12 tok/s, 15-16 steps
- **Inter-step cache**: 1.59x average speedup (9.6 → 15.3 tok/s)
- **6 of 8 prompts outperform llama.cpp** with cache enabled
- **Never slower** than low_confidence on any prompt tested
- **Weighted estimate** on mixed chatbot traffic: ~1.8x speedup
