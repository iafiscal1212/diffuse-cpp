#!/usr/bin/env python3
"""Validate GGUF conversion by comparing tensor values against original SafeTensors.

Loads one tensor at a time from both sources to minimize memory usage.
Reports L2 norm difference and max absolute difference per tensor.

Usage:
    python validate-gguf.py --model /path/to/LLaDA-8B-Instruct --gguf llada-8b-f16.gguf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import gguf
except ImportError:
    print("ERROR: pip install gguf", file=sys.stderr)
    sys.exit(1)

# Same mapping as convert-llada.py
GLOBAL_MAP = {
    "model.transformer.wte.weight":    "token_embd.weight",
    "model.transformer.ln_f.weight":   "output_norm.weight",
    "model.transformer.ff_out.weight": "output.weight",
}
LAYER_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "q_proj.weight":    "attn_q.weight",
    "k_proj.weight":    "attn_k.weight",
    "v_proj.weight":    "attn_v.weight",
    "attn_out.weight":  "attn_output.weight",
    "ff_norm.weight":   "ffn_norm.weight",
    "ff_proj.weight":   "ffn_gate.weight",
    "up_proj.weight":   "ffn_up.weight",
    "ff_out.weight":    "ffn_down.weight",
}


def map_name(name):
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    prefix = "model.transformer.blocks."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        dot = rest.index(".")
        layer_id = rest[:dot]
        component = rest[dot + 1:]
        if component in LAYER_MAP:
            return f"blk.{layer_id}.{LAYER_MAP[component]}"
    return None


def load_gguf_tensor(reader, name):
    """Load a single tensor from GGUF reader as float32 numpy array."""
    for t in reader.tensors:
        if t.name == name:
            data = t.data
            # Dequantize if needed
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif t.tensor_type == gguf.GGMLQuantizationType.F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            else:
                print(f"  WARNING: unsupported quant type {t.tensor_type} for {name}")
                return None
            # GGUF shape is reversed from numpy
            shape = tuple(reversed([int(x) for x in t.shape]))
            return arr.reshape(shape)
    return None


def validate(args):
    model_dir = args.model
    gguf_path = args.gguf

    # Find safetensors files
    shard_files = sorted(Path(model_dir).glob("model*.safetensors"))
    if not shard_files:
        single = Path(model_dir) / "model.safetensors"
        if single.exists():
            shard_files = [single]
        else:
            print(f"No safetensors files in {model_dir}")
            sys.exit(1)

    print(f"Model: {model_dir} ({len(shard_files)} shards)")
    print(f"GGUF:  {gguf_path}")

    # Open GGUF
    reader = gguf.GGUFReader(gguf_path)
    print(f"GGUF tensors: {len(reader.tensors)}")

    n_ok = 0
    n_warn = 0
    n_skip = 0
    max_diff_overall = 0.0
    worst_tensor = ""

    # Compare tensor by tensor
    for shard_path in shard_files:
        print(f"\n── {shard_path.name} ──")
        with safe_open(str(shard_path), framework="pt") as f:
            for name in sorted(f.keys()):
                gguf_name = map_name(name)
                if gguf_name is None:
                    n_skip += 1
                    continue

                # Load original (one at a time) via PyTorch for BF16 support
                import torch
                orig = f.get_tensor(name).float().numpy()

                # Load from GGUF
                gguf_tensor = load_gguf_tensor(reader, gguf_name)
                if gguf_tensor is None:
                    print(f"  MISS  {gguf_name} — not found in GGUF")
                    n_warn += 1
                    continue

                # Compare shapes
                if orig.shape != gguf_tensor.shape:
                    print(f"  SHAPE {name}: orig={orig.shape} vs gguf={gguf_tensor.shape}")
                    n_warn += 1
                    continue

                # Compute differences
                diff = np.abs(orig - gguf_tensor)
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))
                l2_orig = float(np.sqrt(np.sum(orig ** 2)))
                rel_diff = max_diff / (l2_orig / np.sqrt(orig.size) + 1e-10)

                if max_diff_overall < max_diff:
                    max_diff_overall = max_diff
                    worst_tensor = gguf_name

                # F16 conversion introduces error up to ~0.01 relative
                status = "OK" if max_diff < 0.1 else "WARN"
                if status == "WARN":
                    n_warn += 1
                else:
                    n_ok += 1

                if status == "WARN" or args.verbose:
                    print(f"  {status:4s}  {gguf_name:40s}  "
                          f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}  "
                          f"rel={rel_diff:.6f}")
                else:
                    print(f"  {status:4s}  {gguf_name}")

                # Free memory
                del orig, gguf_tensor, diff

    print(f"\n{'='*60}")
    print(f"Results: {n_ok} OK, {n_warn} warnings, {n_skip} skipped")
    print(f"Max absolute difference: {max_diff_overall:.8f} ({worst_tensor})")

    if n_warn == 0:
        print("VALIDATION PASSED — all tensors match within tolerance")
    else:
        print("VALIDATION WARNING — some tensors have large differences")

    return 0 if n_warn == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Validate GGUF against SafeTensors")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to original model directory")
    parser.add_argument("--gguf", "-g", required=True,
                        help="Path to GGUF file to validate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print details for all tensors, not just warnings")
    args = parser.parse_args()
    sys.exit(validate(args))


if __name__ == "__main__":
    main()
