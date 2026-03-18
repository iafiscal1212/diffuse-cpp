#!/usr/bin/env python3
"""End-to-end test: create tiny model, convert, load in C++, run forward pass."""

import json
import os
import sys
import tempfile
import subprocess
import numpy as np

try:
    from safetensors.numpy import save_file
except ImportError:
    print("SKIP: safetensors not installed")
    sys.exit(0)

HIDDEN = 64
N_HEADS = 4
N_LAYERS = 2
VOCAB = 256
FF = 128
MASK_ID = 200

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    build_dir = os.path.join(project_dir, "build")

    test_binary = os.path.join(build_dir, "test-forward")
    if not os.path.exists(test_binary):
        print(f"SKIP: test-forward binary not found at {test_binary}")
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "model")
        os.makedirs(model_dir)

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

        rng = np.random.default_rng(42)
        tensors = {}
        tensors["model.transformer.wte.weight"] = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
        tensors["model.transformer.ln_f.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors["model.transformer.ff_out.weight"] = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)

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

        save_file(tensors, os.path.join(model_dir, "model.safetensors"))

        # Convert to GGUF
        gguf_path = os.path.join(tmpdir, "tiny.gguf")
        converter = os.path.join(project_dir, "tools", "convert-llada.py")

        result = subprocess.run([
            sys.executable, converter,
            "--input", model_dir,
            "--output", gguf_path,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("Converter failed:")
            print(result.stderr)
            sys.exit(1)

        print("Converter OK")

        # Run C++ test-forward with the GGUF file
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{build_dir}:{build_dir}/ggml/src"

        result = subprocess.run(
            [test_binary, gguf_path],
            capture_output=True, text=True, env=env, timeout=30,
        )

        print("C++ test-forward output:")
        print(result.stderr)
        if result.returncode != 0:
            print(f"FAIL: test-forward exited with code {result.returncode}")
            sys.exit(1)

        print("\nEnd-to-end test PASSED!")


if __name__ == "__main__":
    main()
