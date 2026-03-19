#!/usr/bin/env python3
"""Systematic benchmark: entropy_exit speedup across real-world prompt categories.

Measures what fraction of realistic chatbot interactions benefit from entropy_exit.
Runs each prompt with both 'low_confidence' (baseline) and 'entropy_exit' schedulers,
comparing steps used, tok/s, and quality.

Usage:
    python benchmark_entropy_categories.py \
        --model-dir /path/to/LLaDA-8B-Instruct \
        --gguf /path/to/llada-8b-f16.gguf \
        --cpp-bin ./build/diffuse-cli
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ── Prompt categories ──────────────────────────────────────────
PROMPTS = {
    # Category: (prompt, system_prompt_override_or_None)
    "factual_simple": [
        "What is the capital of Japan?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "How many planets are in the solar system?",
    ],
    "factual_complex": [
        "What are the main differences between TCP and UDP?",
        "Explain the three laws of thermodynamics briefly.",
        "What causes tides on Earth?",
        "How does photosynthesis work?",
        "What is the difference between a virus and a bacterium?",
    ],
    "translation": [
        "Translate to French: The weather is beautiful today.",
        "Translate to Spanish: I would like to order a coffee, please.",
        "Translate to German: Where is the nearest train station?",
        "Translate to Japanese: Thank you very much for your help.",
        "Translate to Italian: The book is on the table.",
    ],
    "summarization": [
        "Summarize in one sentence: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems improve their performance through experience.",
        "Summarize in one sentence: The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and other materials, built along the historical northern borders of China to protect against various nomadic groups.",
        "Summarize in one sentence: Quantum computing uses quantum mechanical phenomena such as superposition and entanglement to perform computation. It promises exponential speedups for certain problems compared to classical computers.",
    ],
    "code_simple": [
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to reverse a string.",
        "Write a Python function that calculates the factorial of a number.",
    ],
    "code_complex": [
        "Write a Python function that implements binary search on a sorted list.",
        "Write a Python class for a simple linked list with insert and delete methods.",
        "Write a Python function that finds the longest common subsequence of two strings.",
    ],
    "creative_writing": [
        "Write a haiku about the ocean.",
        "Write a short poem about artificial intelligence.",
        "Write a limerick about a programmer.",
    ],
    "math_reasoning": [
        "Solve step by step: If a train travels at 60 km/h for 2.5 hours, how far does it go?",
        "Solve: What is 15% of 240?",
        "Solve step by step: A rectangle has a perimeter of 30 cm and a width of 5 cm. What is its area?",
    ],
    "classification": [
        "Classify the sentiment of this text as positive, negative, or neutral: I absolutely loved the movie, it was fantastic!",
        "Classify this text as spam or not spam: Congratulations! You've won a free iPhone! Click here to claim.",
        "Extract all named entities from: Albert Einstein was born in Ulm, Germany in 1879 and later moved to Princeton, New Jersey.",
    ],
    "structured_output": [
        "List exactly 5 programming languages and their primary use case, one per line.",
        "List the 4 seasons and one characteristic of each.",
        "List 3 advantages and 3 disadvantages of remote work.",
    ],
    "conversational": [
        "Hello, how are you today?",
        "Can you recommend a good book to read?",
        "What should I cook for dinner tonight?",
    ],
    "instruction_following": [
        "Rewrite this sentence in passive voice: The cat chased the mouse.",
        "Convert this to a question: The Earth revolves around the Sun.",
        "Make this more formal: Hey, can you help me out with this stuff?",
    ],
}

def run_generation(cpp_bin, gguf, tokens_str, n_generate, steps, remasking,
                   entropy_threshold, threads, env):
    """Run diffuse-cli and return (output_tokens_str, elapsed_seconds, stderr)."""
    cmd = [
        cpp_bin,
        "-m", gguf,
        "--tokens", tokens_str,
        "-n", str(n_generate),
        "-s", str(steps),
        "-t", str(threads),
        "--temp", "0.0",
        "--seed", "42",
        "--schedule", "cosine",
        "--remasking", remasking,
        "--entropy-threshold", str(entropy_threshold),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return None, elapsed, result.stderr

    return result.stdout.strip(), elapsed, result.stderr


def count_steps_from_stderr(stderr_text):
    """Parse the actual number of steps from stderr output."""
    # Look for "step X/Y" pattern — the last one tells us how many steps ran
    import re
    steps = re.findall(r'step (\d+)/(\d+)', stderr_text)
    if steps:
        return int(steps[-1][0])  # last step number
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--cpp-bin", default="./build/diffuse-cli")
    parser.add_argument("-n", "--n-generate", type=int, default=64)
    parser.add_argument("-s", "--steps", type=int, default=16)
    parser.add_argument("-t", "--threads", type=int, default=12)
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Only run these categories (default: all)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Setup env
    env = os.environ.copy()
    build_dir = os.path.dirname(os.path.abspath(args.cpp_bin))
    ggml_lib = os.path.join(build_dir, "ggml", "src")
    ld_paths = [build_dir, ggml_lib]
    if "LD_LIBRARY_PATH" in env:
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    categories_to_run = args.categories or list(PROMPTS.keys())
    results = {}
    mask_id = 126336

    total_prompts = sum(len(PROMPTS[c]) for c in categories_to_run)
    prompt_idx = 0

    for category in categories_to_run:
        if category not in PROMPTS:
            print(f"Unknown category: {category}", file=sys.stderr)
            continue

        prompts = PROMPTS[category]
        cat_results = []

        for prompt in prompts:
            prompt_idx += 1
            print(f"\n[{prompt_idx}/{total_prompts}] {category}: {prompt[:60]}...",
                  file=sys.stderr)

            # Tokenize with chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True)
            except Exception:
                text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{prompt}<|im_end|>\n"
                        f"<|im_start|>assistant\n")
                input_ids = tokenizer.encode(text)

            tokens_str = ",".join(map(str, input_ids))
            n_prompt = len(input_ids)

            # Run baseline (low_confidence)
            print(f"  baseline...", file=sys.stderr, end="", flush=True)
            out_base, t_base, stderr_base = run_generation(
                args.cpp_bin, args.gguf, tokens_str,
                args.n_generate, args.steps, "low_confidence",
                args.entropy_threshold, args.threads, env)
            steps_base = count_steps_from_stderr(stderr_base)
            print(f" {t_base:.1f}s ({steps_base} steps)", file=sys.stderr)

            # Run entropy_exit
            print(f"  entropy_exit...", file=sys.stderr, end="", flush=True)
            out_ee, t_ee, stderr_ee = run_generation(
                args.cpp_bin, args.gguf, tokens_str,
                args.n_generate, args.steps, "entropy_exit",
                args.entropy_threshold, args.threads, env)
            steps_ee = count_steps_from_stderr(stderr_ee)
            print(f" {t_ee:.1f}s ({steps_ee} steps)", file=sys.stderr)

            # Decode outputs
            text_base = ""
            text_ee = ""
            if out_base:
                ids = [int(x) for x in out_base.split(",")]
                ids_clean = [t for t in ids if t != mask_id]
                text_base = tokenizer.decode(ids_clean, skip_special_tokens=True)
            if out_ee:
                ids = [int(x) for x in out_ee.split(",")]
                ids_clean = [t for t in ids if t != mask_id]
                text_ee = tokenizer.decode(ids_clean, skip_special_tokens=True)

            speedup = t_base / t_ee if t_ee > 0 else 0
            step_reduction = 1.0 - (steps_ee / steps_base) if steps_base else 0

            cat_results.append({
                "prompt": prompt,
                "n_prompt_tokens": n_prompt,
                "baseline": {
                    "time_s": round(t_base, 2),
                    "steps": steps_base,
                    "text": text_base[:200],
                },
                "entropy_exit": {
                    "time_s": round(t_ee, 2),
                    "steps": steps_ee,
                    "text": text_ee[:200],
                },
                "speedup": round(speedup, 2),
                "step_reduction": round(step_reduction, 3),
            })

            print(f"  speedup: {speedup:.2f}x, steps: {steps_base}→{steps_ee} "
                  f"({step_reduction*100:.0f}% reduction)", file=sys.stderr)

        results[category] = cat_results

    # Summary
    print("\n" + "="*80, file=sys.stderr)
    print("SUMMARY BY CATEGORY", file=sys.stderr)
    print("="*80, file=sys.stderr)

    total_with_speedup = 0
    total_prompts_run = 0
    category_summaries = {}

    for category, cat_results in results.items():
        speedups = [r["speedup"] for r in cat_results]
        step_reds = [r["step_reduction"] for r in cat_results]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        avg_step_red = sum(step_reds) / len(step_reds) if step_reds else 0
        has_speedup = sum(1 for s in speedups if s > 1.2)  # >20% faster

        total_with_speedup += has_speedup
        total_prompts_run += len(cat_results)

        category_summaries[category] = {
            "n_prompts": len(cat_results),
            "avg_speedup": round(avg_speedup, 2),
            "avg_step_reduction": round(avg_step_red, 3),
            "prompts_with_speedup": has_speedup,
        }

        tag = "***" if avg_speedup > 1.5 else "**" if avg_speedup > 1.2 else ""
        print(f"  {category:25s}: avg {avg_speedup:.2f}x speedup, "
              f"avg {avg_step_red*100:.0f}% step reduction, "
              f"{has_speedup}/{len(cat_results)} with >20% speedup {tag}",
              file=sys.stderr)

    pct_benefit = (total_with_speedup / total_prompts_run * 100
                   if total_prompts_run > 0 else 0)
    print(f"\n  TOTAL: {total_with_speedup}/{total_prompts_run} prompts "
          f"({pct_benefit:.0f}%) benefit from entropy_exit (>20% speedup)",
          file=sys.stderr)

    # Save full results
    output = {
        "config": {
            "n_generate": args.n_generate,
            "steps": args.steps,
            "threads": args.threads,
            "entropy_threshold": args.entropy_threshold,
        },
        "category_summaries": category_summaries,
        "total_with_speedup": total_with_speedup,
        "total_prompts": total_prompts_run,
        "pct_benefit": round(pct_benefit, 1),
        "detailed_results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
