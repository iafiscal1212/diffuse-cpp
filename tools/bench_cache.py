#!/usr/bin/env python3
"""Benchmark 8 real prompts: cache vs no-cache, Q4_K_M, entropy_exit, n=256."""
import subprocess, sys, json, os, re

MODEL = "/root/models/llada-8b-q4km.gguf"
CLI = "/root/diffuse-cpp/build/diffuse-cli"
N_GEN = 256
STEPS = 16
THREADS = 12
SEED = 42

PROMPTS = [
    "What is the capital of France?",
    "Translate to French: The weather is beautiful today",
    "Write a Python function to check if a number is prime",
    "Write a short poem about the ocean",
    "Explain why the sky is blue in two sentences",
    "List the planets in our solar system",
    "What is 15 multiplied by 23?",
    "Translate to Spanish: I love programming",
]

from transformers import AutoTokenizer
print("Loading tokenizer...", file=sys.stderr, flush=True)
tok = AutoTokenizer.from_pretrained("/root/models/LLaDA-8B-Instruct", trust_remote_code=True)

all_tokens = []
for p in PROMPTS:
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    all_tokens.append(",".join(map(str, ids)))

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/root/diffuse-cpp/build:/root/diffuse-cpp/build/ggml/src"
mask_id = 126336


def run_test(tokens_str, cache_mode):
    cmd = [CLI, "-m", MODEL, "--tokens", tokens_str,
           "-n", str(N_GEN), "-s", str(STEPS), "-t", str(THREADS),
           "--remasking", "entropy_exit", "--seed", str(SEED)]
    if cache_mode == "no-cache":
        cmd.append("--no-cache")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("  ERROR: " + result.stderr[:200], file=sys.stderr, flush=True)
        return 0, 0, "ERROR", 0
    toks = 0.0
    total_ms = 0.0
    m = re.search(r"TOTAL:\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)", result.stderr)
    if m:
        total_ms = float(m.group(1))
        toks = float(m.group(2))
    actual_steps = len(re.findall(r"step \d+/\d+: unmasked", result.stderr))
    out_ids = [int(x) for x in result.stdout.strip().split(",") if x.strip()]
    clean = [x for x in out_ids if x != mask_id]
    text = tok.decode(clean, skip_special_tokens=True)
    return toks, total_ms, text, actual_steps


results = []
for mode in ["cache", "no-cache"]:
    print("\n" + "=" * 60, file=sys.stderr, flush=True)
    print("  MODE: " + mode, file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    for i, (prompt, tokens_str) in enumerate(zip(PROMPTS, all_tokens)):
        label = prompt[:50]
        print("  [%d/8] %s..." % (i + 1, label), file=sys.stderr, flush=True)
        toks, ms, text, steps = run_test(tokens_str, mode)
        results.append({
            "prompt_idx": i + 1,
            "prompt": prompt,
            "mode": mode,
            "tok_s": round(toks, 2),
            "ms": round(ms, 1),
            "text": text[:500],
            "steps": steps,
        })
        print("         %.2f tok/s | %.0fms | %d steps" % (toks, ms, steps),
              file=sys.stderr, flush=True)

print(json.dumps(results, indent=2))
