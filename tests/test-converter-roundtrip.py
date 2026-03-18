#!/usr/bin/env python3
"""Smoke test: create a tiny dummy LLaDA model, convert to GGUF, verify it loads."""

import json
import os
import sys
import tempfile
import numpy as np

try:
    from safetensors.numpy import save_file
    import gguf
except ImportError:
    print("SKIP: safetensors or gguf not installed")
    sys.exit(0)

# Tiny model params
HIDDEN = 64
N_HEADS = 4
N_LAYERS = 2
VOCAB = 256
FF = 128  # intermediate_size
MASK_ID = 200

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "model")
        os.makedirs(model_dir)

        # Write config.json
        config = {
            "hidden_size": HIDDEN,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": N_HEADS,
            "num_hidden_layers": N_LAYERS,
            "intermediate_size": FF,
            "vocab_size": VOCAB,
            "max_position_embeddings": 512,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "mask_token_id": MASK_ID,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Create dummy tensors
        rng = np.random.default_rng(42)
        tensors = {}

        # Global tensors
        tensors["model.transformer.wte.weight"] = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
        tensors["model.transformer.ln_f.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors["model.transformer.ff_out.weight"] = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)

        # Per-layer tensors
        for i in range(N_LAYERS):
            p = f"model.transformer.blocks.{i}"
            tensors[f"{p}.attn_norm.weight"] = np.ones(HIDDEN, dtype=np.float32)
            tensors[f"{p}.q_proj.weight"]    = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32)
            tensors[f"{p}.k_proj.weight"]    = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32)
            tensors[f"{p}.v_proj.weight"]    = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32)
            tensors[f"{p}.attn_out.weight"]  = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32)
            tensors[f"{p}.ff_norm.weight"]   = np.ones(HIDDEN, dtype=np.float32)
            tensors[f"{p}.ff_proj.weight"]   = rng.standard_normal((FF, HIDDEN)).astype(np.float32)
            tensors[f"{p}.up_proj.weight"]   = rng.standard_normal((FF, HIDDEN)).astype(np.float32)
            tensors[f"{p}.ff_out.weight"]    = rng.standard_normal((HIDDEN, FF)).astype(np.float32)

        # Save as single safetensors file
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))

        # Run converter
        gguf_path = os.path.join(tmpdir, "test.gguf")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../tools")
        from importlib import import_module
        import subprocess

        result = subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "tools", "convert-llada.py"),
            "--input", model_dir,
            "--output", gguf_path,
            "--type", "f32",
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            print("FAIL: converter returned non-zero exit code")
            sys.exit(1)

        # Verify GGUF file
        assert os.path.exists(gguf_path), "GGUF file not created"
        size = os.path.getsize(gguf_path)
        assert size > 0, "GGUF file is empty"

        # Read back with gguf library
        reader = gguf.GGUFReader(gguf_path)

        # Check metadata
        expected_tensors = 3 + N_LAYERS * 9  # 3 global + 9 per layer
        n_tensors = len(reader.tensors)
        assert n_tensors == expected_tensors, \
            f"Expected {expected_tensors} tensors, got {n_tensors}"

        print(f"\nRoundtrip test PASSED!")
        print(f"  GGUF size: {size:,} bytes")
        print(f"  Tensors: {n_tensors}")
        print(f"  Metadata fields: {len(reader.fields)}")

        # Print all metadata
        for name in reader.fields:
            if not name.startswith("GGUF."):
                field = reader.fields[name]
                print(f"    {name}")

        # Print tensor names
        for t in reader.tensors:
            print(f"    tensor: {t.name}  shape={list(t.shape)}  type={t.tensor_type.name}")


if __name__ == "__main__":
    main()
