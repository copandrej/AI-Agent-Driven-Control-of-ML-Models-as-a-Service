#!/usr/bin/env bash
# Run the evaluation grid for ONE model × all archs × N repetitions.
#
# Usage:
#   bash eval/run_grid.sh <model> [runs] [--cluster microk8s]
#
# Models:  qwen | minimax | qwen9b
# Cluster: kind (default) | microk8s
#
# Examples:
#   bash eval/run_grid.sh qwen 2
#   bash eval/run_grid.sh qwen9b 2 --cluster microk8s
set -euo pipefail

MODEL_KEY=${1:-}
RUNS=${2:-2}
CLUSTER="kind"

# parse --cluster flag
for arg in "$@"; do
    if [[ "$arg" == "--cluster" ]]; then
        CLUSTER_NEXT=1
    elif [[ "${CLUSTER_NEXT:-0}" == "1" ]]; then
        CLUSTER="$arg"
        CLUSTER_NEXT=0
    fi
done

case "$MODEL_KEY" in
    qwen)
        MODEL_NAME="Qwen-Qwen3.5-122B-A10B"
        MODEL_ENV="LLM_PROVIDER=openai_compat OPENAI_COMPAT_MODEL=$MODEL_NAME"
        ;;
    minimax)
        MODEL_NAME="MiniMaxAI-MiniMax-M2.5"
        MODEL_ENV="LLM_PROVIDER=openai_compat OPENAI_COMPAT_MODEL=$MODEL_NAME"
        ;;
    qwen9b)
        MODEL_NAME="qwen3.5:9b-64k"
        MODEL_ENV="LLM_PROVIDER=ollama OLLAMA_MODEL=$MODEL_NAME"
        ;;
    *)
        echo "Usage: bash eval/run_grid.sh <model> [runs] [--cluster microk8s]"
        echo "  models: qwen | minimax | qwen9b"
        exit 1
        ;;
esac

# cluster-specific setup
if [[ "$CLUSTER" == "microk8s" ]]; then
    RESULTS_DIR="eval/results/microk8s"
    export KUBECONFIG=/tmp/microk8s.yaml
    export PROMETHEUS_URL=http://193.2.205.27:30090
    export GRAFANA_URL=http://193.2.205.27:30030
    # refresh kubeconfig from VM in case it expired
    ssh acop@193.2.205.27 "microk8s config" > /tmp/microk8s.yaml
    echo "Cluster: microk8s  (KUBECONFIG=/tmp/microk8s.yaml)"
else
    RESULTS_DIR="eval/results"
    echo "Cluster: kind"
fi

echo "Model  : $MODEL_NAME"
echo "Archs  : flow simple minimal dual"
echo "Runs   : $RUNS"
echo "Results: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

for ((i=1; i<=RUNS; i++)); do
    echo "════ Run $i/$RUNS ════"
    env $MODEL_ENV EVAL_RESULTS_DIR="$RESULTS_DIR" \
        uv run python eval/runner.py --archs flow simple minimal dual
done

echo ""
echo "Done. Aggregating..."
uv run python eval/aggregate.py "$RESULTS_DIR"
