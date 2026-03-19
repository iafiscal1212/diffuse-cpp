## Table 1: Performance by Quantization (steps=16, threads=12)

| Model | Size (GB) | Scheduler | Time (ms) | tok/s | Steps | Speedup vs F16 |
|---|---|---|---|---|---|---|
| F16 | 14.9 | low_confidence | 38962 | 1.64 | 16 | 1.00x |
| F16 | 14.9 | entropy_exit | 7333 | 8.74 | 3 | 5.32x |
| Q8_0 | 8.4 | low_confidence | 34403 | 1.86 | 16 | 1.13x |
| Q8_0 | 8.4 | entropy_exit | 6339 | 10.09 | 3 | 6.14x |
| Q4_K_M | 5.1 | low_confidence | 25783 | 2.48 | 16 | 1.51x |
| Q4_K_M | 5.1 | entropy_exit | 4709 | 13.59 | 3 | 8.27x |

## Table 2: Thread Scaling (Q4_K_M, steps=16)

| Threads | Scheduler | Time (ms) | tok/s | Scaling vs t=1 |
|---|---|---|---|---|
| 1 | low_confidence | 190740 | 0.34 | 1.0x |
| 4 | low_confidence | 54465 | 1.17 | 3.5x |
| 12 | low_confidence | 25783 | 2.48 | 7.4x |
| 24 | low_confidence | 31327 | 2.05 | 6.1x |
| 1 | entropy_exit | 36050 | 1.78 | 1.0x |
| 4 | entropy_exit | 10082 | 6.35 | 3.6x |
| 12 | entropy_exit | 4709 | 13.59 | 7.6x |
| 24 | entropy_exit | 5631 | 11.38 | 6.4x |

## Table 3: Diffusion Steps vs Throughput (Q4_K_M, threads=12)

| Steps | Scheduler | Time (ms) | tok/s | Actual Steps |
|---|---|---|---|---|
| 8 | low_confidence | 12579 | 5.09 | 8 |
| 16 | low_confidence | 25783 | 2.48 | 16 |
| 32 | low_confidence | 51724 | 1.24 | 32 |
| 8 | entropy_exit | 5073 | 12.75 | 3 |
| 16 | entropy_exit | 4709 | 13.59 | 3 |
| 32 | entropy_exit | 5052 | 12.76 | 3 |

## Table 4: Full Benchmark Matrix

| Model | Steps | Threads | Scheduler | Time (ms) | tok/s |
|---|---|---|---|---|---|
| F16 | 8 | 4 | low_confidence | 36429 | 1.76 |
| F16 | 8 | 12 | low_confidence | 19245 | 3.33 |
| F16 | 8 | 24 | low_confidence | 22785 | 2.81 |
| F16 | 16 | 4 | low_confidence | 73944 | 0.87 |
| F16 | 16 | 12 | low_confidence | 38962 | 1.64 |
| F16 | 16 | 24 | low_confidence | 45908 | 1.39 |
| F16 | 32 | 4 | low_confidence | 147915 | 0.43 |
| F16 | 32 | 12 | low_confidence | 78010 | 0.82 |
| F16 | 32 | 24 | low_confidence | 87717 | 0.73 |
| F16 | 8 | 4 | entropy_exit | 14326 | 4.47 |
| F16 | 8 | 12 | entropy_exit | 7129 | 8.98 |
| F16 | 8 | 24 | entropy_exit | 8120 | 7.92 |
| F16 | 16 | 4 | entropy_exit | 14163 | 4.52 |
| F16 | 16 | 12 | entropy_exit | 7333 | 8.74 |
| F16 | 16 | 24 | entropy_exit | 8179 | 7.84 |
| F16 | 32 | 4 | entropy_exit | 14169 | 4.52 |
| F16 | 32 | 12 | entropy_exit | 7151 | 8.95 |
| F16 | 32 | 24 | entropy_exit | 8577 | 7.50 |
| Q8_0 | 8 | 4 | low_confidence | 37848 | 1.69 |
| Q8_0 | 8 | 12 | low_confidence | 17222 | 3.72 |
| Q8_0 | 8 | 24 | low_confidence | 17838 | 3.59 |
| Q8_0 | 16 | 4 | low_confidence | 75679 | 0.85 |
| Q8_0 | 16 | 12 | low_confidence | 34403 | 1.86 |
| Q8_0 | 16 | 24 | low_confidence | 35613 | 1.80 |
| Q8_0 | 32 | 4 | low_confidence | 151480 | 0.42 |
| Q8_0 | 32 | 12 | low_confidence | 68103 | 0.94 |
| Q8_0 | 32 | 24 | low_confidence | 72111 | 0.89 |
| Q8_0 | 8 | 4 | entropy_exit | 14610 | 4.38 |
| Q8_0 | 8 | 12 | entropy_exit | 6352 | 10.07 |
| Q8_0 | 8 | 24 | entropy_exit | 6693 | 9.58 |
| Q8_0 | 16 | 4 | entropy_exit | 14725 | 4.35 |
| Q8_0 | 16 | 12 | entropy_exit | 6339 | 10.09 |
| Q8_0 | 16 | 24 | entropy_exit | 6582 | 9.73 |
| Q8_0 | 32 | 4 | entropy_exit | 14486 | 4.42 |
| Q8_0 | 32 | 12 | entropy_exit | 6340 | 10.10 |
| Q8_0 | 32 | 24 | entropy_exit | 6714 | 9.53 |
| Q4_K_M | 8 | 1 | low_confidence | 96657 | 0.66 |
| Q4_K_M | 8 | 4 | low_confidence | 26949 | 2.37 |
| Q4_K_M | 8 | 12 | low_confidence | 12579 | 5.09 |
| Q4_K_M | 8 | 24 | low_confidence | 15481 | 4.15 |
| Q4_K_M | 16 | 1 | low_confidence | 190740 | 0.34 |
| Q4_K_M | 16 | 4 | low_confidence | 54465 | 1.17 |
| Q4_K_M | 16 | 12 | low_confidence | 25783 | 2.48 |
| Q4_K_M | 16 | 24 | low_confidence | 31327 | 2.05 |
| Q4_K_M | 32 | 1 | low_confidence | 379442 | 0.17 |
| Q4_K_M | 32 | 4 | low_confidence | 108844 | 0.59 |
| Q4_K_M | 32 | 12 | low_confidence | 51724 | 1.24 |
| Q4_K_M | 32 | 24 | low_confidence | 61068 | 1.05 |
| Q4_K_M | 8 | 1 | entropy_exit | 36164 | 1.77 |
| Q4_K_M | 8 | 4 | entropy_exit | 9977 | 6.41 |
| Q4_K_M | 8 | 12 | entropy_exit | 5073 | 12.75 |
| Q4_K_M | 8 | 24 | entropy_exit | 5802 | 11.08 |
| Q4_K_M | 16 | 1 | entropy_exit | 36050 | 1.78 |
| Q4_K_M | 16 | 4 | entropy_exit | 10082 | 6.35 |
| Q4_K_M | 16 | 12 | entropy_exit | 4709 | 13.59 |
| Q4_K_M | 16 | 24 | entropy_exit | 5631 | 11.38 |
| Q4_K_M | 32 | 1 | entropy_exit | 36088 | 1.77 |
| Q4_K_M | 32 | 4 | entropy_exit | 10038 | 6.38 |
| Q4_K_M | 32 | 12 | entropy_exit | 5052 | 12.76 |
| Q4_K_M | 32 | 24 | entropy_exit | 5623 | 11.40 |

## Summary

- **Best throughput**: 13.59 tok/s (Q4_K_M s=16 t=12 entropy_exit)
- **F16 baseline** (s=16, t=12, low_confidence): 1.64 tok/s
- **Best speedup vs F16 baseline**: 8.3x
