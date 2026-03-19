#!/usr/bin/env python3
"""Format benchmark JSON results into publication-ready tables.

Usage:
    python format-benchmark.py research/benchmark/bench_*.json
"""

import json
import sys
import os
from collections import defaultdict


def load_results(paths):
    """Load all benchmark JSON files."""
    all_results = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        # Infer model label from filename
        basename = os.path.basename(path)
        label = basename.replace("bench_", "").replace(".json", "")
        all_results[label] = data
    return all_results


def compute_averages(results_list, n_reps):
    """Group results by (scheduler, steps, threads) and average."""
    groups = defaultdict(list)
    for r in results_list:
        key = (r["scheduler"], r["steps"], r["threads"])
        groups[key].append(r)

    averages = {}
    for key, entries in groups.items():
        avg_ms = sum(e["elapsed_ms"] for e in entries) / len(entries)
        avg_tps = sum(e["tok_per_sec"] for e in entries) / len(entries)
        actual_steps = entries[-1].get("actual_steps", key[1])
        averages[key] = {
            "elapsed_ms": avg_ms,
            "tok_per_sec": avg_tps,
            "actual_steps": actual_steps,
        }
    return averages


def main():
    if len(sys.argv) < 2:
        print("Usage: python format-benchmark.py bench_*.json")
        sys.exit(1)

    all_data = load_results(sys.argv[1:])

    # ── Table 1: Performance by quantization (fixed steps=16, threads=12) ──
    print("## Table 1: Performance by Quantization (steps=16, threads=12)")
    print()
    print("| Model | Size (GB) | Scheduler | Time (ms) | tok/s | Steps | Speedup vs F16 |")
    print("|---|---|---|---|---|---|---|")

    sizes = {"F16": "14.9", "Q8_0": "8.4", "Q4_K_M": "5.1"}
    model_order = ["F16", "Q8_0", "Q4_K_M"]

    f16_baseline_tps = None
    for model in model_order:
        if model not in all_data:
            continue
        data = all_data[model]
        avgs = compute_averages(data["results"], data["n_reps"])

        for sched in ["low_confidence", "entropy_exit"]:
            key = (sched, 16, 12)
            if key not in avgs:
                continue
            r = avgs[key]
            if model == "F16" and sched == "low_confidence":
                f16_baseline_tps = r["tok_per_sec"]

            speedup = ""
            if f16_baseline_tps and f16_baseline_tps > 0:
                speedup = f"{r['tok_per_sec'] / f16_baseline_tps:.2f}x"

            print(f"| {model} | {sizes.get(model, '?')} | {sched} | "
                  f"{r['elapsed_ms']:.0f} | {r['tok_per_sec']:.2f} | "
                  f"{r['actual_steps']} | {speedup} |")

    # ── Table 2: Thread scaling (Q4_K_M, steps=16) ──
    print()
    print("## Table 2: Thread Scaling (Q4_K_M, steps=16)")
    print()
    print("| Threads | Scheduler | Time (ms) | tok/s | Scaling vs t=1 |")
    print("|---|---|---|---|---|")

    if "Q4_K_M" in all_data:
        data = all_data["Q4_K_M"]
        avgs = compute_averages(data["results"], data["n_reps"])

        base_tps = {}
        for sched in ["low_confidence", "entropy_exit"]:
            key1 = (sched, 16, 1)
            if key1 in avgs:
                base_tps[sched] = avgs[key1]["tok_per_sec"]

        for sched in ["low_confidence", "entropy_exit"]:
            for t in [1, 4, 12, 24]:
                key = (sched, 16, t)
                if key not in avgs:
                    continue
                r = avgs[key]
                scaling = ""
                if sched in base_tps and base_tps[sched] > 0:
                    scaling = f"{r['tok_per_sec'] / base_tps[sched]:.1f}x"
                print(f"| {t} | {sched} | {r['elapsed_ms']:.0f} | "
                      f"{r['tok_per_sec']:.2f} | {scaling} |")

    # ── Table 3: Steps vs quality (Q4_K_M, threads=12) ──
    print()
    print("## Table 3: Diffusion Steps vs Throughput (Q4_K_M, threads=12)")
    print()
    print("| Steps | Scheduler | Time (ms) | tok/s | Actual Steps |")
    print("|---|---|---|---|---|")

    if "Q4_K_M" in all_data:
        data = all_data["Q4_K_M"]
        avgs = compute_averages(data["results"], data["n_reps"])

        for sched in ["low_confidence", "entropy_exit"]:
            for s in [8, 16, 32]:
                key = (sched, s, 12)
                if key not in avgs:
                    continue
                r = avgs[key]
                print(f"| {s} | {sched} | {r['elapsed_ms']:.0f} | "
                      f"{r['tok_per_sec']:.2f} | {r['actual_steps']} |")

    # ── Table 4: Full matrix (all models, key configs) ──
    print()
    print("## Table 4: Full Benchmark Matrix")
    print()
    print("| Model | Steps | Threads | Scheduler | Time (ms) | tok/s |")
    print("|---|---|---|---|---|---|")

    for model in model_order:
        if model not in all_data:
            continue
        data = all_data[model]
        avgs = compute_averages(data["results"], data["n_reps"])

        for sched in ["low_confidence", "entropy_exit"]:
            for s in [8, 16, 32]:
                for t in sorted(set(r["threads"] for r in data["results"])):
                    key = (sched, s, t)
                    if key not in avgs:
                        continue
                    r = avgs[key]
                    print(f"| {model} | {s} | {t} | {sched} | "
                          f"{r['elapsed_ms']:.0f} | {r['tok_per_sec']:.2f} |")

    # ── Summary ──
    print()
    print("## Summary")
    print()

    # Best config
    best_tps = 0
    best_config = ""
    for model in model_order:
        if model not in all_data:
            continue
        data = all_data[model]
        avgs = compute_averages(data["results"], data["n_reps"])
        for key, r in avgs.items():
            if r["tok_per_sec"] > best_tps:
                best_tps = r["tok_per_sec"]
                best_config = f"{model} s={key[1]} t={key[2]} {key[0]}"

    print(f"- **Best throughput**: {best_tps:.2f} tok/s ({best_config})")

    if f16_baseline_tps:
        print(f"- **F16 baseline** (s=16, t=12, low_confidence): {f16_baseline_tps:.2f} tok/s")
        print(f"- **Best speedup vs F16 baseline**: {best_tps / f16_baseline_tps:.1f}x")


if __name__ == "__main__":
    main()
