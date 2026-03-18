#!/usr/bin/env python3
"""Convert LLaDA SafeTensors model to GGUF format for diffuse-cpp.

LLaDA-8B uses a Llama-like backbone with bidirectional attention.
Tensor names follow the OLMo convention (model.transformer.blocks.{i}.*).

Usage:
    python convert-llada.py --input /path/to/LLaDA-8B-Instruct --output llada-8b-f32.gguf
    python convert-llada.py --input /path/to/LLaDA-8B-Instruct --output llada-8b-f16.gguf --type f16
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import gguf
except ImportError:
    print("ERROR: gguf not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)


# ── Tensor name mapping ────────────────────────────────────────
# LLaDA SafeTensors → diffuse-cpp GGUF

GLOBAL_TENSOR_MAP = {
    "model.transformer.wte.weight":    "token_embd.weight",
    "model.transformer.ln_f.weight":   "output_norm.weight",
    "model.transformer.ff_out.weight": "output.weight",       # lm_head
}

# Per-layer: model.transformer.blocks.{i}.X → blk.{i}.Y
LAYER_TENSOR_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "q_proj.weight":    "attn_q.weight",
    "k_proj.weight":    "attn_k.weight",
    "v_proj.weight":    "attn_v.weight",
    "attn_out.weight":  "attn_output.weight",
    "ff_norm.weight":   "ffn_norm.weight",
    "ff_proj.weight":   "ffn_gate.weight",   # SwiGLU gate (w1)
    "up_proj.weight":   "ffn_up.weight",     # SwiGLU up (w3)
    "ff_out.weight":    "ffn_down.weight",   # SwiGLU down (w2)
}

# Expected tensor shapes for LLaDA-8B (for validation)
EXPECTED_SHAPES_8B = {
    "token_embd.weight":        (126464, 4096),
    "output_norm.weight":       (4096,),
    "output.weight":            (126464, 4096),
    "blk.*.attn_norm.weight":   (4096,),
    "blk.*.attn_q.weight":      (4096, 4096),
    "blk.*.attn_k.weight":      (4096, 4096),
    "blk.*.attn_v.weight":      (4096, 4096),
    "blk.*.attn_output.weight": (4096, 4096),
    "blk.*.ffn_norm.weight":    (4096,),
    "blk.*.ffn_gate.weight":    (12288, 4096),
    "blk.*.ffn_up.weight":      (12288, 4096),
    "blk.*.ffn_down.weight":    (4096, 12288),
}


def map_tensor_name(name: str) -> str:
    """Map LLaDA tensor name to GGUF tensor name. Returns None if unmapped."""
    if name in GLOBAL_TENSOR_MAP:
        return GLOBAL_TENSOR_MAP[name]

    prefix = "model.transformer.blocks."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        dot_idx = rest.index(".")
        layer_id = rest[:dot_idx]
        component = rest[dot_idx + 1:]
        if component in LAYER_TENSOR_MAP:
            return f"blk.{layer_id}.{LAYER_TENSOR_MAP[component]}"

    return None


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    u32 = raw.view(np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def load_config(model_dir: str) -> dict:
    """Load config.json from model directory."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def find_shard_files(model_dir: str) -> list:
    """Find all SafeTensors shard files, sorted."""
    files = sorted(Path(model_dir).glob("model*.safetensors"))
    if not files:
        # Try single-file format
        single = Path(model_dir) / "model.safetensors"
        if single.exists():
            return [str(single)]
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return [str(f) for f in files]


def validate_shape(gguf_name: str, shape: tuple, n_layers: int):
    """Validate tensor shape against expected shapes."""
    # Try exact match first
    if gguf_name in EXPECTED_SHAPES_8B:
        expected = EXPECTED_SHAPES_8B[gguf_name]
        if shape != expected:
            print(f"  WARNING: {gguf_name} shape {shape} != expected {expected}")
            return False
        return True

    # Try wildcard match for layer tensors
    for pattern, expected in EXPECTED_SHAPES_8B.items():
        if "*" in pattern:
            # blk.*.X → blk.{any_number}.X
            prefix, suffix = pattern.split("*")
            if gguf_name.startswith(prefix) and gguf_name.endswith(suffix):
                if shape != expected:
                    print(f"  WARNING: {gguf_name} shape {shape} != expected {expected}")
                    return False
                return True

    return True  # Unknown tensor, skip validation


def convert(args):
    """Main conversion logic."""
    config = load_config(args.input)
    shard_files = find_shard_files(args.input)

    print(f"Model directory: {args.input}")
    print(f"Shard files: {len(shard_files)}")
    print(f"Output: {args.output}")
    print(f"Output type: {args.type}")

    # ── Extract hyperparameters from config.json ────────────────
    # LLaDA uses various config key names; handle alternatives
    hidden_size    = config.get("hidden_size",    config.get("d_model", 4096))
    n_heads        = config.get("num_attention_heads", config.get("n_heads", 32))
    n_kv_heads     = config.get("num_key_value_heads", config.get("n_kv_heads", n_heads))
    n_layers       = config.get("num_hidden_layers", config.get("n_layers", 32))
    intermediate   = config.get("intermediate_size", config.get("mlp_hidden_size", 12288))
    vocab_size     = config.get("vocab_size", 126464)
    max_pos        = config.get("max_position_embeddings",
                                config.get("max_sequence_length", 4096))
    rope_theta     = float(config.get("rope_theta", 500000.0))
    rms_norm_eps   = float(config.get("rms_norm_eps",
                                       config.get("layer_norm_eps", 1e-5)))
    mask_token_id  = config.get("mask_token_id", 126336)

    print(f"\nHyperparameters (from config.json):")
    print(f"  hidden_size      = {hidden_size}")
    print(f"  n_heads          = {n_heads}")
    print(f"  n_kv_heads       = {n_kv_heads}")
    print(f"  n_layers         = {n_layers}")
    print(f"  intermediate     = {intermediate}")
    print(f"  vocab_size       = {vocab_size}")
    print(f"  max_pos          = {max_pos}")
    print(f"  rope_theta       = {rope_theta}")
    print(f"  rms_norm_eps     = {rms_norm_eps}")
    print(f"  mask_token_id    = {mask_token_id}")

    # ── Create GGUF writer ──────────────────────────────────────
    writer = gguf.GGUFWriter(args.output, arch="diffuse")

    # Use typed GGUF methods → standard key names:
    #   diffuse.block_count, diffuse.attention.head_count, etc.
    writer.add_name("LLaDA-8B")
    writer.add_block_count(n_layers)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate)
    writer.add_context_length(max_pos)
    writer.add_vocab_size(vocab_size)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_norm_eps)

    # Custom diffusion-specific keys
    writer.add_uint32("diffuse.mask_token_id", mask_token_id)
    writer.add_string("diffuse.model_type", "llada")

    # ── Process tensors ─────────────────────────────────────────
    n_converted = 0
    n_skipped = 0
    total_params = 0
    checksums = {}

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path)
        print(f"\n── {shard_name} ──")

        with safe_open(shard_path, framework="numpy") as f:
            for name in sorted(f.keys()):
                gguf_name = map_tensor_name(name)
                if gguf_name is None:
                    print(f"  SKIP  {name}")
                    n_skipped += 1
                    continue

                tensor = f.get_tensor(name)
                orig_dtype = tensor.dtype
                orig_shape = tensor.shape

                # Handle bfloat16 (numpy doesn't support it natively)
                if orig_dtype == np.dtype("uint16"):
                    # safetensors may return bf16 as uint16
                    tensor = bf16_to_f32(tensor)
                    orig_dtype_str = "bf16"
                elif hasattr(orig_dtype, "name") and "bfloat" in orig_dtype.name:
                    tensor = tensor.view(np.uint16).astype(np.uint32)
                    tensor = (tensor << 16).view(np.float32)
                    orig_dtype_str = "bf16"
                else:
                    orig_dtype_str = str(orig_dtype)

                # Convert to output type
                if args.type == "f32":
                    tensor = tensor.astype(np.float32)
                elif args.type == "f16":
                    tensor = tensor.astype(np.float16)
                # bf16: keep as f32 and let GGUF handle it

                # Validate shape
                validate_shape(gguf_name, orig_shape, n_layers)

                # Compute checksum for validation
                if args.validate:
                    f32_data = tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor
                    checksum = float(np.sum(f32_data ** 2))
                    checksums[gguf_name] = checksum

                n_params = int(np.prod(orig_shape))
                total_params += n_params

                print(f"  OK    {name}")
                print(f"        → {gguf_name}  {list(orig_shape)}  "
                      f"{orig_dtype_str} → {args.type}  ({n_params:,} params)")

                writer.add_tensor(gguf_name, tensor)
                n_converted += 1

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Tensors converted: {n_converted}")
    print(f"Tensors skipped:   {n_skipped}")
    print(f"Total parameters:  {total_params:,} ({total_params/1e9:.2f}B)")

    expected_tensors = 3 + n_layers * 9  # 3 global + 9 per layer
    if n_converted != expected_tensors:
        print(f"WARNING: expected {expected_tensors} tensors, got {n_converted}")

    # ── Write GGUF ──────────────────────────────────────────────
    print(f"\nWriting GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_bytes = os.path.getsize(args.output)
    size_gb = size_bytes / (1024 ** 3)
    print(f"Output: {args.output} ({size_gb:.2f} GB)")

    # ── Validation checksums ────────────────────────────────────
    if args.validate:
        cksum_path = args.output + ".checksums.json"
        with open(cksum_path, "w") as f:
            json.dump(checksums, f, indent=2)
        print(f"Checksums written to: {cksum_path}")
        print("To validate against PyTorch:")
        print(f"  python -c \"import torch, json; ...")
        print(f"  # Compare sum(tensor**2) for each tensor\"")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaDA SafeTensors model to GGUF format for diffuse-cpp")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to LLaDA model directory "
                             "(containing config.json and .safetensors files)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output GGUF file path")
    parser.add_argument("--type", "-t", default="f32",
                        choices=["f32", "f16"],
                        help="Output tensor type (default: f32)")
    parser.add_argument("--validate", action="store_true",
                        help="Write tensor checksums for cross-validation with PyTorch")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert(args)


if __name__ == "__main__":
    main()
