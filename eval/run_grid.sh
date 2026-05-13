#!/usr/bin/env bash
# Run the evaluation grid for ONE HPC model × all archs × N repetitions.
# Load the model on HPC, run tunnel.sh, then call this script.
#
# Usage:
#   bash eval/run_grid.sh <model> [RUNS]
#
# Models:
#   qwen      → Qwen-Qwen3.5-122B-A10B
#   minimax   → MiniMaxAI-MiniMax-M2.5
#   qwen9b    → Qwen-Qwen3.5-9B  (via Ollama)
#
# Examples:
#   bash eval/run_grid.sh qwen        # 2 runs (default)
#   bash eval/run_grid.sh minimax 5   # 5 runs
set -euo pipefail

MODEL_KEY=${1:-}
RUNS=${2:-2}
ARCHS=(flow simple minimal dual)

case "$MODEL_KEY" in
    qwen)
        MODEL_NAME="Qwen-Qwen3.5-122B-A10B"
        EXTRA_ENV="LLM_PROVIDER=openai_compat OPENAI_COMPAT_MODEL=$MODEL_NAME"
        ;;
    minimax)
        MODEL_NAME="MiniMaxAI-MiniMax-M2.5"
        EXTRA_ENV="LLM_PROVIDER=openai_compat OPENAI_COMPAT_MODEL=$MODEL_NAME"
        ;;
    qwen9b)
        MODEL_NAME="qwen3.5:9b-64k"
        EXTRA_ENV="LLM_PROVIDER=ollama OLLAMA_MODEL=$MODEL_NAME"
        ;;
    *)
        echo "Usage: bash eval/run_grid.sh <model> [runs]"
        echo "  models: qwen | minimax | qwen9b"
        exit 1
        ;;
esac

echo "Model : $MODEL_NAME"
echo "Archs : ${ARCHS[*]}"
echo "Runs  : $RUNS"
echo "Output: eval/results/"
echo ""

for ((i=1; i<=RUNS; i++)); do
    echo "════ Run $i/$RUNS ════"
    env LLM_PROVIDER=openai_compat OPENAI_COMPAT_MODEL="$MODEL_NAME" \
        uv run python eval/runner.py --archs "${ARCHS[@]}"
done

echo ""
echo "Done. Aggregating..."
uv run python eval/aggregate.py eval/results/
