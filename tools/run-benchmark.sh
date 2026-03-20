#!/bin/bash
# Full benchmark suite for diffuse-cpp publication
# Runs F16, Q8_0, Q4_K_M across steps/threads/schedulers
#
# Usage: ./tools/run-benchmark.sh [--tokens TOKEN_IDS]
#
# Requires: models in ./models/ (or $MODEL_DIR)

set -e

MODEL_DIR="${MODEL_DIR:-./models}"
BENCH="./build/diffuse-bench"
RESULTS_DIR="./research/benchmark"

STEPS="8,16,32"
THREADS="1,4,12,24"
REPS=3
N_GEN=256
WARMUP=1

# Default tokens: pre-tokenized "What is the capital of France?" with chat template
# (LLaDA-8B-Instruct chat format)
TOKENS="${1:-126080,126346,18621,126347,198,198,2496,449,259,9031,16841,13,126348,126346,3840,126347,198,198,2372,341,268,7706,300,11406,30,126348,126346,598,10450,126347,198,198}"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "diffuse-cpp Benchmark Suite"
echo "============================================================"
echo "Steps: $STEPS"
echo "Threads: $THREADS"
echo "Reps: $REPS"
echo "Generate: $N_GEN tokens"
echo ""

MODELS=(
    "llada-8b-f16.gguf:F16"
    "llada-8b-q8_0.gguf:Q8_0"
    "llada-8b-q4km.gguf:Q4_K_M"
)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_file label <<< "$entry"
    model_path="$MODEL_DIR/$model_file"

    if [ ! -f "$model_path" ]; then
        echo "SKIP: $model_path not found"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Model: $label ($model_file)"
    echo "  Size: $(du -h "$model_path" | cut -f1)"
    echo "============================================================"

    # JSON output
    $BENCH -m "$model_path" \
        --tokens "$TOKENS" \
        -n $N_GEN \
        -s "$STEPS" \
        -t "$THREADS" \
        -r $REPS \
        --warmup $WARMUP \
        --json \
        > "$RESULTS_DIR/bench_${label}.json" \
        2> >(tee "$RESULTS_DIR/bench_${label}.log" >&2)

    # Markdown table
    $BENCH -m "$model_path" \
        --tokens "$TOKENS" \
        -n $N_GEN \
        -s "$STEPS" \
        -t "$THREADS" \
        -r $REPS \
        --warmup $WARMUP \
        > "$RESULTS_DIR/bench_${label}.md" \
        2>/dev/null

    echo ""
    cat "$RESULTS_DIR/bench_${label}.md"
done

echo ""
echo "============================================================"
echo "Results saved to $RESULTS_DIR/"
echo "============================================================"
ls -la "$RESULTS_DIR/"
