#!/usr/bin/env python3
"""End-to-end text generation with diffuse-cpp.

Handles tokenization (via HuggingFace) and calls the C++ binary.

Usage:
    python generate.py \
        --model-dir /path/to/LLaDA-8B-Instruct \
        --gguf llada-8b-f16.gguf \
        --cpp-bin ./build/diffuse-cli \
        -p "Explain quantum computing in one sentence."
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate text with diffuse-cpp")
    parser.add_argument("--model-dir", "-m", required=True,
                        help="HuggingFace model directory (for tokenizer)")
    parser.add_argument("--gguf", "-g", required=True, help="GGUF model file")
    parser.add_argument("--cpp-bin", default="./build/diffuse-cli",
                        help="Path to diffuse-cli binary")
    parser.add_argument("-p", "--prompt", required=True, help="User prompt")
    parser.add_argument("-n", "--n-generate", type=int, default=256,
                        help="Tokens to generate (default: 256)")
    parser.add_argument("-s", "--steps", type=int, default=32,
                        help="Diffusion steps (default: 32)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="CPU threads (default: 4)")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Temperature (default: 0 = argmax)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--schedule", default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--remasking", default="low_confidence",
                        choices=["low_confidence", "random", "entropy_exit",
                                 "maskgit_plus", "topk_margin"])
    parser.add_argument("--entropy-threshold", type=float, default=1.5,
                        help="Entropy threshold for entropy_exit scheduler")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable inter-step KV cache")
    parser.add_argument("--cache-refresh", type=int, default=0,
                        help="Force full forward every N steps (default: 0 = never)")
    parser.add_argument("--cache-keep-active", type=int, default=0,
                        help="Keep recently-changed positions active N extra steps (default: 0)")
    parser.add_argument("--system", default="You are a helpful assistant.",
                        help="System prompt")
    parser.add_argument("--raw", action="store_true",
                        help="Don't apply chat template, tokenize prompt directly")
    args = parser.parse_args()

    # --- Tokenize ---
    from transformers import AutoTokenizer

    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if args.raw:
        input_ids = tokenizer.encode(args.prompt)
    else:
        # Apply chat template
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        # Try apply_chat_template first
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
        except Exception:
            # Fallback: manual template
            text = (f"<|im_start|>system\n{args.system}<|im_end|>\n"
                    f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
            input_ids = tokenizer.encode(text)

    print(f"Prompt: {len(input_ids)} tokens", file=sys.stderr)
    print(f"Generating: {args.n_generate} tokens, {args.steps} steps", file=sys.stderr)

    tokens_str = ",".join(map(str, input_ids))

    # --- Run C++ binary ---
    env = os.environ.copy()
    # Add build dirs to LD_LIBRARY_PATH
    build_dir = os.path.dirname(os.path.abspath(args.cpp_bin))
    ggml_lib = os.path.join(build_dir, "ggml", "src")
    ld_paths = [build_dir, ggml_lib]
    if "LD_LIBRARY_PATH" in env:
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    cmd = [
        args.cpp_bin,
        "-m", args.gguf,
        "--tokens", tokens_str,
        "-n", str(args.n_generate),
        "-s", str(args.steps),
        "-t", str(args.threads),
        "--temp", str(args.temp),
        "--seed", str(args.seed),
        "--schedule", args.schedule,
        "--remasking", args.remasking,
        "--entropy-threshold", str(args.entropy_threshold),
    ]

    if args.no_cache:
        cmd.append("--no-cache")
    if args.cache_refresh > 0:
        cmd.extend(["--cache-refresh", str(args.cache_refresh)])
    if args.cache_keep_active > 0:
        cmd.extend(["--cache-keep-active", str(args.cache_keep_active)])

    print(f"Running: {' '.join(cmd[:6])}...", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("C++ binary failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Print stderr (progress) to stderr
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    # Parse output token IDs
    output_line = result.stdout.strip()
    if not output_line:
        print("No output from C++ binary", file=sys.stderr)
        sys.exit(1)

    output_ids = [int(x) for x in output_line.split(",")]

    # --- Detokenize ---
    # Filter out mask tokens and special tokens
    mask_id = 126336
    output_ids_clean = [t for t in output_ids if t != mask_id]

    output_text = tokenizer.decode(output_ids_clean, skip_special_tokens=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Generated {len(output_ids)} tokens "
          f"({len(output_ids_clean)} non-mask):", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(output_text)


if __name__ == "__main__":
    main()
