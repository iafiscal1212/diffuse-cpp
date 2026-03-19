#!/usr/bin/env python3
"""Semantic scheduling experiments for diffusion LLMs.

Compares uniform vs entropy-guided unmasking schedules on LLaDA-8B.
Measures: quality, speed, entropy distribution.

Usage:
    python semantic_scheduler.py --model /path/to/LLaDA-8B-Instruct \
        --prompts prompts.txt --n-generate 64
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Schedulers ──────────────────────────────────────────────────

class UniformScheduler:
    """Baseline: unmask N/steps tokens per step, ordered by confidence."""
    name = "uniform"

    def __init__(self, n_steps):
        self.n_steps = n_steps

    def get_unmask_count(self, step, n_masked):
        """How many tokens to unmask this step."""
        remaining_steps = self.n_steps - step
        if remaining_steps <= 0:
            return n_masked
        return max(1, round(n_masked / remaining_steps))

    def select_to_unmask(self, candidates, n_unmask, step):
        """Select which candidates to unmask (sorted by confidence desc)."""
        sorted_c = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
        return sorted_c[:n_unmask]


class CrystallizationScheduler:
    """Phase-based: low entropy first (scaffold), then medium, then high."""
    name = "crystallization"

    def __init__(self, n_steps, low_thresh=1.5, high_thresh=3.5):
        self.n_steps = n_steps
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def get_unmask_count(self, step, n_masked):
        remaining_steps = self.n_steps - step
        if remaining_steps <= 0:
            return n_masked
        return max(1, round(n_masked / remaining_steps))

    def select_to_unmask(self, candidates, n_unmask, step):
        """Unmask by entropy group: low first, then medium, then high."""
        # Sort by entropy ascending (easiest first)
        sorted_c = sorted(candidates, key=lambda c: c["entropy"])
        return sorted_c[:n_unmask]


class EarlyExitScheduler:
    """Standard schedule but unmask MORE tokens when they're easy (low entropy)."""
    name = "early_exit"

    def __init__(self, n_steps, easy_thresh=1.5, easy_multiplier=2.0):
        self.n_steps = n_steps
        self.easy_thresh = easy_thresh
        self.easy_multiplier = easy_multiplier

    def get_unmask_count(self, step, n_masked):
        remaining_steps = self.n_steps - step
        if remaining_steps <= 0:
            return n_masked
        return max(1, round(n_masked / remaining_steps))

    def select_to_unmask(self, candidates, n_unmask, step):
        """Unmask all easy tokens + scheduled amount of harder tokens."""
        sorted_c = sorted(candidates, key=lambda c: c["entropy"])

        # Count how many are "easy" (low entropy)
        n_easy = sum(1 for c in sorted_c if c["entropy"] < self.easy_thresh)

        # Unmask at least n_unmask, but also all easy tokens
        n_actual = max(n_unmask, min(n_easy, len(sorted_c)))
        return sorted_c[:n_actual]


class AdaptiveScheduler:
    """Pure entropy-ordered: always unmask lowest entropy first."""
    name = "adaptive"

    def __init__(self, n_steps):
        self.n_steps = n_steps

    def get_unmask_count(self, step, n_masked):
        remaining_steps = self.n_steps - step
        if remaining_steps <= 0:
            return n_masked
        # Cosine schedule (more at start)
        t0 = step / self.n_steps
        t1 = (step + 1) / self.n_steps
        cos0 = math.cos(t0 * math.pi * 0.5)
        cos1 = math.cos(t1 * math.pi * 0.5)
        frac = (cos0 - cos1) / (cos0 + 1e-10)
        n = max(1, round(frac * n_masked))
        return min(n, n_masked)

    def select_to_unmask(self, candidates, n_unmask, step):
        """Always unmask lowest entropy positions first."""
        sorted_c = sorted(candidates, key=lambda c: c["entropy"])
        return sorted_c[:n_unmask]


# ── Diffusion sampling engine ──────────────────────────────────

@dataclass
class StepStats:
    step: int
    n_unmasked: int
    n_remaining: int
    entropies: list = field(default_factory=list)
    mean_entropy: float = 0.0
    time_ms: float = 0.0


def compute_entropy(logits, vocab_size):
    """Compute entropy of softmax distribution."""
    # Numerically stable softmax
    max_logit = logits.max()
    exp_logits = torch.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()

    # Entropy: -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def generate_with_scheduler(model, tokenizer, prompt, n_generate, scheduler,
                            mask_id=126336, device="cpu"):
    """Run diffusion generation with a given scheduler."""
    # Tokenize with chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
    except Exception:
        input_ids = tokenizer.encode(prompt)

    prompt_len = len(input_ids)

    # Build sequence: prompt + masks
    seq = input_ids + [mask_id] * n_generate
    total_len = len(seq)

    is_masked = [False] * prompt_len + [True] * n_generate
    n_masked = n_generate

    step_stats = []

    total_forward_time = 0.0

    for step in range(scheduler.n_steps):
        if n_masked == 0:
            break

        # Forward pass
        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        forward_ms = (time.time() - t0) * 1000
        total_forward_time += forward_ms

        # Build candidates for masked positions
        candidates = []
        entropies = []
        for i in range(total_len):
            if not is_masked[i]:
                continue

            pos_logits = logits[i].float()
            entropy = compute_entropy(pos_logits, logits.shape[-1])
            entropies.append(entropy)

            # Argmax token and confidence
            token_id = pos_logits.argmax().item()
            confidence = pos_logits.max().item()

            candidates.append({
                "pos": i,
                "token": token_id,
                "confidence": confidence,
                "entropy": entropy,
            })

        # Determine how many to unmask
        n_unmask = scheduler.get_unmask_count(step, n_masked)

        # Select which to unmask
        to_unmask = scheduler.select_to_unmask(candidates, n_unmask, step)

        # Apply unmasking
        for c in to_unmask:
            seq[c["pos"]] = c["token"]
            is_masked[c["pos"]] = False
            n_masked -= 1

        stats = StepStats(
            step=step,
            n_unmasked=len(to_unmask),
            n_remaining=n_masked,
            entropies=entropies,
            mean_entropy=np.mean(entropies) if entropies else 0.0,
            time_ms=forward_ms,
        )
        step_stats.append(stats)

    # Decode result
    output_ids = seq[prompt_len:]
    output_ids_clean = [t for t in output_ids if t != mask_id]
    output_text = tokenizer.decode(output_ids_clean, skip_special_tokens=True)

    return {
        "text": output_text,
        "output_ids": output_ids,
        "n_forward_passes": len(step_stats),
        "total_forward_ms": total_forward_time,
        "effective_steps": len(step_stats),
        "step_stats": step_stats,
        "tok_per_s": n_generate / (total_forward_time / 1000) if total_forward_time > 0 else 0,
    }


# ── Main experiment ─────────────────────────────────────────────

def run_experiments(args):
    print("Loading model...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Translate to French: The weather is beautiful today.",
        "Write a short poem about the ocean.",
        "What are the three laws of thermodynamics?",
    ]

    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]

    n_generate = args.n_generate
    n_steps = args.n_steps

    schedulers = [
        UniformScheduler(n_steps),
        CrystallizationScheduler(n_steps),
        EarlyExitScheduler(n_steps),
        AdaptiveScheduler(n_steps),
    ]

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx+1}: {prompt}")
        print(f"{'='*60}")

        for sched in schedulers:
            print(f"\n  [{sched.name}] ", end="", flush=True)

            result = generate_with_scheduler(
                model, tokenizer, prompt, n_generate, sched)

            print(f"{result['effective_steps']} steps, "
                  f"{result['total_forward_ms']:.0f}ms, "
                  f"{result['tok_per_s']:.1f} tok/s")
            print(f"    Output: {result['text'][:100]}...")

            # Entropy analysis from first step
            if result["step_stats"]:
                first_step = result["step_stats"][0]
                entropies = first_step.entropies
                if entropies:
                    ent_arr = np.array(entropies)
                    n_low = np.sum(ent_arr < 1.5)
                    n_mid = np.sum((ent_arr >= 1.5) & (ent_arr < 3.5))
                    n_high = np.sum(ent_arr >= 3.5)
                    print(f"    Step-1 entropy: mean={np.mean(ent_arr):.2f}, "
                          f"low(<1.5)={n_low}, mid={n_mid}, high(>3.5)={n_high}")

            results.append({
                "prompt": prompt,
                "scheduler": sched.name,
                "n_steps": n_steps,
                "n_generate": n_generate,
                "effective_steps": result["effective_steps"],
                "total_forward_ms": result["total_forward_ms"],
                "tok_per_s": result["tok_per_s"],
                "text": result["text"],
                "entropy_evolution": [
                    {"step": s.step, "mean_entropy": s.mean_entropy,
                     "n_unmasked": s.n_unmasked, "n_remaining": s.n_remaining}
                    for s in result["step_stats"]
                ],
                "step1_entropies": result["step_stats"][0].entropies if result["step_stats"] else [],
            })

    # Save results
    output_path = args.output or "fase4.2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Scheduler':<20s} {'Steps':>6s} {'Time(ms)':>10s} {'tok/s':>8s} {'Text preview'}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['scheduler']:<20s} {r['effective_steps']:>6d} "
              f"{r['total_forward_ms']:>10.0f} {r['tok_per_s']:>8.1f} "
              f"{r['text'][:35]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--prompts", help="File with prompts, one per line")
    parser.add_argument("--n-generate", "-n", type=int, default=64)
    parser.add_argument("--n-steps", "-s", type=int, default=16)
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
