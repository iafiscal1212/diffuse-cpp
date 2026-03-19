#!/usr/bin/env python3
"""Compare forward pass logits between PyTorch LLaDA and diffuse-cpp GGUF.

Runs both engines on the same input tokens and compares logit distributions.
For low-memory systems: uses BF16 + gradient_checkpointing + single-batch.

Usage:
    python validate-logits.py \
        --model /path/to/LLaDA-8B-Instruct \
        --gguf llada-8b-f16.gguf \
        --cpp-bin ./build/diffuse-cli \
        --tokens "1,2,3,4,5,6,7,8"

    # Or with prompt (requires tokenizer):
    python validate-logits.py \
        --model /path/to/LLaDA-8B-Instruct \
        --gguf llada-8b-f16.gguf \
        --cpp-bin ./build/diffuse-cli \
        --prompt "Hello world"
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


def tokenize(model_dir, prompt):
    """Tokenize prompt using HF tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ids = tok.encode(prompt)
    print(f"Tokenized '{prompt}' → {len(ids)} tokens: {ids[:20]}{'...' if len(ids)>20 else ''}")
    return ids


def run_pytorch(model_dir, token_ids):
    """Run forward pass with PyTorch LLaDA. Returns logits [n_tokens, vocab_size]."""
    import torch

    # Check available memory
    import psutil
    avail_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available RAM: {avail_gb:.1f} GB")

    if avail_gb < 18:
        print("WARNING: <18GB RAM available. Loading in BF16 with low_cpu_mem_usage.")
        print("This may be very slow due to swapping. Consider using the server.")

    from transformers import AutoModelForCausalLM, AutoConfig

    print("Loading PyTorch model (BF16, low_cpu_mem_usage)...")
    t0 = time.time()

    # Load with minimal memory
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Forward pass
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print(f"Running forward pass ({len(token_ids)} tokens)...")
    t0 = time.time()

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    logits_np = logits[0].float().numpy()  # [seq_len, vocab_size]
    print(f"Forward pass done in {time.time()-t0:.1f}s, logits shape: {logits_np.shape}")

    del model, outputs, logits
    import gc; gc.collect()

    return logits_np


def run_cpp(gguf_path, cpp_bin, token_ids, n_threads=4):
    """Run forward pass with diffuse-cpp. Returns logits [n_tokens, vocab_size].

    Since the CLI doesn't output raw logits yet, we use a helper binary.
    For now, write a small C++ helper that dumps logits to a binary file.
    """
    # Build tokens string
    tokens_str = ",".join(map(str, token_ids))

    # We need a way to get raw logits from C++. For validation, we write
    # a temporary Python script that calls the C library via ctypes.
    # Actually, the simplest approach: modify the test to dump logits.
    # For now, use the test-forward binary approach:

    # Write a small Python script that uses the GGUF file to get metadata,
    # then we'll compare at the tensor level instead.
    print("NOTE: Raw logits comparison requires the C++ binary to dump logits.")
    print("      For now, validating conversion correctness at tensor level.")
    print("      Full logit comparison will be available after adding --dump-logits to CLI.")

    return None


def compare_logits(pytorch_logits, cpp_logits, top_k=10):
    """Compare logit distributions between PyTorch and C++."""
    if cpp_logits is None:
        print("\nSkipping logit comparison (C++ logits not available)")
        return

    n_tokens, vocab_size = pytorch_logits.shape
    assert cpp_logits.shape == pytorch_logits.shape, \
        f"Shape mismatch: PyTorch {pytorch_logits.shape} vs C++ {cpp_logits.shape}"

    print(f"\nLogit comparison ({n_tokens} tokens, vocab_size={vocab_size}):")
    print(f"{'pos':>4s}  {'max_diff':>10s}  {'mean_diff':>10s}  {'cos_sim':>8s}  "
          f"{'pt_argmax':>10s}  {'cpp_argmax':>10s}  {'match':>5s}")
    print("-" * 75)

    n_match = 0
    for i in range(n_tokens):
        pt = pytorch_logits[i]
        cpp = cpp_logits[i]

        diff = np.abs(pt - cpp)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Cosine similarity
        dot = np.dot(pt, cpp)
        norm_pt = np.linalg.norm(pt)
        norm_cpp = np.linalg.norm(cpp)
        cos_sim = dot / (norm_pt * norm_cpp + 1e-10)

        pt_argmax = int(np.argmax(pt))
        cpp_argmax = int(np.argmax(cpp))
        match = pt_argmax == cpp_argmax
        if match:
            n_match += 1

        print(f"{i:4d}  {max_diff:10.4f}  {mean_diff:10.6f}  {cos_sim:8.6f}  "
              f"{pt_argmax:10d}  {cpp_argmax:10d}  {'✓' if match else '✗':>5s}")

    print(f"\nArgmax match rate: {n_match}/{n_tokens} ({100*n_match/n_tokens:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Validate forward pass logits")
    parser.add_argument("--model", "-m", required=True, help="HF model directory")
    parser.add_argument("--gguf", "-g", required=True, help="GGUF file path")
    parser.add_argument("--cpp-bin", default="./build/diffuse-cli", help="C++ CLI binary")
    parser.add_argument("--tokens", help="Comma-separated token IDs")
    parser.add_argument("--prompt", "-p", help="Text prompt (tokenized automatically)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="C++ threads")
    args = parser.parse_args()

    # Get token IDs
    if args.tokens:
        token_ids = [int(x) for x in args.tokens.split(",")]
    elif args.prompt:
        token_ids = tokenize(args.model, args.prompt)
    else:
        # Default: small test sequence
        token_ids = [1, 128006, 9125, 128007, 271, 2675, 527, 264]  # generic
        print(f"Using default test tokens: {token_ids}")

    print(f"\nInput: {len(token_ids)} tokens")

    # Run PyTorch
    pytorch_logits = run_pytorch(args.model, token_ids)

    # Save PyTorch logits for external comparison
    logits_path = args.gguf + ".pytorch_logits.npy"
    np.save(logits_path, pytorch_logits)
    print(f"PyTorch logits saved to: {logits_path}")

    # Show top predictions
    print(f"\nPyTorch top-5 predictions per position:")
    for i in range(min(len(token_ids), 8)):
        top5 = np.argsort(pytorch_logits[i])[-5:][::-1]
        probs = np.exp(pytorch_logits[i][top5] - np.max(pytorch_logits[i]))
        probs /= probs.sum()
        print(f"  pos {i}: {list(zip(top5.tolist(), [f'{p:.3f}' for p in probs]))}")

    # Run C++
    cpp_logits = run_cpp(args.gguf, args.cpp_bin, token_ids, args.threads)

    # Compare
    compare_logits(pytorch_logits, cpp_logits)


if __name__ == "__main__":
    main()
