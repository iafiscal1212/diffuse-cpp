#!/usr/bin/env python3
"""Compare C++ logits (from dump-logits) against PyTorch reference.

Usage:
    # Step 1: Generate PyTorch reference logits
    python compare-logits.py --gen-ref \
        --model /path/to/LLaDA-8B-Instruct \
        --tokens 1,2,3,4,5,6,7,8 \
        -o ref_logits.bin

    # Step 2: Generate C++ logits (via dump-logits binary)
    ./build/dump-logits -m llada-8b-f16.gguf --tokens 1,2,3,4,5,6,7,8 -o cpp_logits.bin

    # Step 3: Compare
    python compare-logits.py --compare ref_logits.bin cpp_logits.bin
"""

import argparse
import struct
import sys

import numpy as np


def read_logits_bin(path):
    """Read logits binary file: [int32 n_tokens, int32 n_vocab, float32[] data]."""
    with open(path, "rb") as f:
        n_tokens, n_vocab = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
        assert data.size == n_tokens * n_vocab, \
            f"Expected {n_tokens*n_vocab} floats, got {data.size}"
        return data.reshape(n_tokens, n_vocab)


def gen_reference(model_dir, token_ids, output_path):
    """Generate reference logits from PyTorch model."""
    import torch
    print(f"Loading model from {model_dir}...")

    # Try to load with trust_remote_code for LLaDA custom model
    from transformers import AutoModelForCausalLM
    import time

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print(f"Forward pass: {len(token_ids)} tokens...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0].float().numpy()  # [seq, vocab]
    print(f"Done in {time.time()-t0:.1f}s, shape={logits.shape}")

    # Save in same binary format as dump-logits
    n_tokens, n_vocab = logits.shape
    with open(output_path, "wb") as f:
        f.write(struct.pack("ii", n_tokens, n_vocab))
        f.write(logits.tobytes())

    print(f"Saved: {output_path} ({logits.nbytes/(1024**2):.1f} MB)")

    # Print top-5
    for i in range(min(n_tokens, 8)):
        top5 = np.argsort(logits[i])[-5:][::-1]
        print(f"  pos {i} (tok={token_ids[i]}): {list(top5)}")

    del model
    return logits


def compare(ref_path, cpp_path):
    """Compare two logits binary files."""
    ref = read_logits_bin(ref_path)
    cpp = read_logits_bin(cpp_path)

    print(f"Reference: {ref_path} shape={ref.shape}")
    print(f"C++:       {cpp_path} shape={cpp.shape}")

    assert ref.shape == cpp.shape, \
        f"Shape mismatch: ref={ref.shape} vs cpp={cpp.shape}"

    n_tokens, n_vocab = ref.shape
    print(f"\n{'pos':>4s}  {'max_diff':>10s}  {'mean_diff':>10s}  {'cos_sim':>8s}  "
          f"{'ref_top1':>10s}  {'cpp_top1':>10s}  {'match':>5s}")
    print("-" * 70)

    n_match = 0
    max_diff_all = 0.0

    for i in range(n_tokens):
        r = ref[i]
        c = cpp[i]

        diff = np.abs(r - c)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        max_diff_all = max(max_diff_all, max_diff)

        cos_sim = float(np.dot(r, c) / (np.linalg.norm(r) * np.linalg.norm(c) + 1e-10))

        ref_top = int(np.argmax(r))
        cpp_top = int(np.argmax(c))
        match = ref_top == cpp_top
        if match:
            n_match += 1

        print(f"{i:4d}  {max_diff:10.4f}  {mean_diff:10.6f}  {cos_sim:8.6f}  "
              f"{ref_top:10d}  {cpp_top:10d}  {'OK' if match else 'MISS':>5s}")

    print(f"\n{'='*70}")
    print(f"Argmax match: {n_match}/{n_tokens} ({100*n_match/n_tokens:.1f}%)")
    print(f"Max absolute diff: {max_diff_all:.6f}")

    # For F16 conversion, max_diff up to ~0.5 is normal; cosine sim > 0.99 is good
    if n_match == n_tokens:
        print("RESULT: PERFECT — all top-1 predictions match")
    elif n_match / n_tokens >= 0.9:
        print("RESULT: GOOD — >90% top-1 match (expected with F16 quantization)")
    else:
        print("RESULT: POOR — significant divergence, check conversion")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen-ref", help="Generate PyTorch reference logits")
    gen.add_argument("--model", "-m", required=True)
    gen.add_argument("--tokens", required=True, help="Comma-separated token IDs")
    gen.add_argument("-o", "--output", default="ref_logits.bin")

    cmp = sub.add_parser("compare", help="Compare two logits files")
    cmp.add_argument("ref", help="Reference logits file")
    cmp.add_argument("cpp", help="C++ logits file")

    args = parser.parse_args()

    if args.cmd == "gen-ref":
        tokens = [int(x) for x in args.tokens.split(",")]
        gen_reference(args.model, tokens, args.output)
    elif args.cmd == "compare":
        compare(args.ref, args.cpp)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
