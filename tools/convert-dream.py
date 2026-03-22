#!/usr/bin/env python3
"""Convert Dream SafeTensors model to GGUF format for diffuse-cpp.

Dream-7B uses a Qwen2.5 backbone with bidirectional attention and GQA.
Tensor names follow the HuggingFace Qwen2 convention (model.layers.{i}.*).

Usage:
    python convert-dream.py --input /path/to/Dream-v0-Instruct-7B --output dream-7b-f16.gguf
    python convert-dream.py --input /path/to/Dream-v0-Instruct-7B --output dream-7b-f16.gguf --type f16
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)

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
# Dream (Qwen2.5) SafeTensors → diffuse-cpp GGUF

GLOBAL_TENSOR_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight":         "output_norm.weight",
    "lm_head.weight":            "output.weight",
}

# Per-layer: model.layers.{i}.X → blk.{i}.Y
LAYER_TENSOR_MAP = {
    "self_attn.q_proj.weight":            "attn_q.weight",
    "self_attn.k_proj.weight":            "attn_k.weight",
    "self_attn.v_proj.weight":            "attn_v.weight",
    "self_attn.o_proj.weight":            "attn_output.weight",
    "mlp.gate_proj.weight":               "ffn_gate.weight",   # SwiGLU gate (w1)
    "mlp.up_proj.weight":                 "ffn_up.weight",     # SwiGLU up (w3)
    "mlp.down_proj.weight":               "ffn_down.weight",   # SwiGLU down (w2)
    "input_layernorm.weight":             "attn_norm.weight",
    "post_attention_layernorm.weight":    "ffn_norm.weight",
}

# Expected tensor shapes for Dream-7B (for validation)
EXPECTED_SHAPES_7B = {
    "token_embd.weight":        (152064, 3584),
    "output_norm.weight":       (3584,),
    "output.weight":            (152064, 3584),
    "blk.*.attn_norm.weight":   (3584,),
    "blk.*.attn_q.weight":      (3584, 3584),      # 28 heads * 128
    "blk.*.attn_k.weight":      (512, 3584),        # 4 KV heads * 128
    "blk.*.attn_v.weight":      (512, 3584),        # 4 KV heads * 128
    "blk.*.attn_output.weight": (3584, 3584),
    "blk.*.ffn_norm.weight":    (3584,),
    "blk.*.ffn_gate.weight":    (18944, 3584),
    "blk.*.ffn_up.weight":      (18944, 3584),
    "blk.*.ffn_down.weight":    (3584, 18944),
}


def map_tensor_name(name: str) -> str:
    """Map Dream tensor name to GGUF tensor name. Returns None if unmapped."""
    if name in GLOBAL_TENSOR_MAP:
        return GLOBAL_TENSOR_MAP[name]

    prefix = "model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        dot_idx = rest.index(".")
        layer_id = rest[:dot_idx]
        component = rest[dot_idx + 1:]
        if component in LAYER_TENSOR_MAP:
            return f"blk.{layer_id}.{LAYER_TENSOR_MAP[component]}"

    return None


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
        single = Path(model_dir) / "model.safetensors"
        if single.exists():
            return [str(single)]
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return [str(f) for f in files]


def validate_shape(gguf_name: str, shape: tuple, n_layers: int):
    """Validate tensor shape against expected shapes."""
    if gguf_name in EXPECTED_SHAPES_7B:
        expected = EXPECTED_SHAPES_7B[gguf_name]
        if shape != expected:
            print(f"  WARNING: {gguf_name} shape {shape} != expected {expected}")
            return False
        return True

    for pattern, expected in EXPECTED_SHAPES_7B.items():
        if "*" in pattern:
            prefix, suffix = pattern.split("*")
            if gguf_name.startswith(prefix) and gguf_name.endswith(suffix):
                if shape != expected:
                    print(f"  WARNING: {gguf_name} shape {shape} != expected {expected}")
                    return False
                return True

    return True


def convert(args):
    """Main conversion logic."""
    config = load_config(args.input)
    shard_files = find_shard_files(args.input)

    print(f"Model directory: {args.input}")
    print(f"Shard files: {len(shard_files)}")
    print(f"Output: {args.output}")
    print(f"Output type: {args.type}")

    # ── Extract hyperparameters from config.json ────────────────
    hidden_size    = config.get("hidden_size", 3584)
    n_heads        = config.get("num_attention_heads", 28)
    n_kv_heads     = config.get("num_key_value_heads", 4)
    n_layers       = config.get("num_hidden_layers", 28)
    intermediate   = config.get("intermediate_size", 18944)
    vocab_size     = config.get("vocab_size", 152064)
    max_pos        = config.get("max_position_embeddings", 131072)
    rope_theta     = float(config.get("rope_theta", 1000000.0))
    rms_norm_eps   = float(config.get("rms_norm_eps", 1e-6))
    mask_token_id  = config.get("mask_token_id", 151666)
    tie_embeddings = config.get("tie_word_embeddings", False)

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
    print(f"  tie_embeddings   = {tie_embeddings}")

    # ── Create GGUF writer ──────────────────────────────────────
    writer = gguf.GGUFWriter(args.output, arch="diffuse")

    writer.add_name("Dream-7B")
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
    writer.add_string("diffuse.model_type", "dream")

    # ── Process tensors ─────────────────────────────────────────
    n_converted = 0
    n_skipped = 0
    total_params = 0
    checksums = {}
    has_lm_head = False

    for shard_path in shard_files:
        shard_name = os.path.basename(shard_path)
        print(f"\n── {shard_name} ──")

        with safe_open(shard_path, framework="pt") as f:
            for name in sorted(f.keys()):
                gguf_name = map_tensor_name(name)
                if gguf_name is None:
                    print(f"  SKIP  {name}")
                    n_skipped += 1
                    continue

                if gguf_name == "output.weight":
                    has_lm_head = True

                pt_tensor = f.get_tensor(name)
                orig_dtype_str = str(pt_tensor.dtype).replace("torch.", "")
                orig_shape = tuple(pt_tensor.shape)

                tensor = pt_tensor.float().numpy()

                if args.type == "f32":
                    tensor = tensor.astype(np.float32)
                elif args.type == "f16":
                    tensor = tensor.astype(np.float16)

                validate_shape(gguf_name, orig_shape, n_layers)

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

    # Warn if lm_head not found and tie_embeddings is False
    if not has_lm_head and not tie_embeddings:
        print("WARNING: lm_head.weight not found and tie_word_embeddings=False")
        print("         The model will use token_embd.weight as output projection")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Tensors converted: {n_converted}")
    print(f"Tensors skipped:   {n_skipped}")
    print(f"Total parameters:  {total_params:,} ({total_params/1e9:.2f}B)")

    expected_tensors = 3 + n_layers * 9  # 3 global + 9 per layer
    if not has_lm_head:
        expected_tensors -= 1  # No separate lm_head
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

    if args.validate:
        cksum_path = args.output + ".checksums.json"
        with open(cksum_path, "w") as f:
            json.dump(checksums, f, indent=2)
        print(f"Checksums written to: {cksum_path}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Dream SafeTensors model to GGUF format for diffuse-cpp")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to Dream model directory "
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
