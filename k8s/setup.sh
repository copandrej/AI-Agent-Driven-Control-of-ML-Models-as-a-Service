#!/usr/bin/env bash
# Full Kind cluster bootstrap: creates cluster, builds/loads images,
# installs Prometheus/Grafana, deploys all services.
set -euo pipefail

CLUSTER_NAME="agent-cluster"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Prerequisite checks ──────────────────────────────────────────────────

for cmd in kind kubectl docker helm; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' is required but not found in PATH."
        exit 1
    fi
done

# ── Cluster ──────────────────────────────────────────────────────────────

if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "Deleting existing cluster '${CLUSTER_NAME}'..."
    kind delete cluster --name "$CLUSTER_NAME"
fi

echo "Creating Kind cluster '${CLUSTER_NAME}'..."
kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml"

# ── Docker images ────────────────────────────────────────────────────────

echo ""
echo "=== Building Docker Images ==="

# Same names as on Docker Hub, so the manifests work on Kind (local images) and other clusters (pull)
echo "Building copandrej/echo-service:latest..."
docker build -t copandrej/echo-service:latest "$SCRIPT_DIR/echo_service"

echo "Building copandrej/ml-service:v1 (mobilenet_v2) and :v2 (resnet50)..."
docker build -t copandrej/ml-service:v1 "$SCRIPT_DIR/ml_service"
docker build -t copandrej/ml-service:v2 --build-arg MODEL_NAME=resnet50 --build-arg MODEL_VERSION=v2 "$SCRIPT_DIR/ml_service"

echo "Loading images into Kind..."
kind load docker-image copandrej/echo-service:latest copandrej/ml-service:v1 copandrej/ml-service:v2 --name "$CLUSTER_NAME"

# ── Prometheus + Grafana ─────────────────────────────────────────────────

echo ""
echo "=== Installing Prometheus Stack ==="

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

# Pinned for reproducibility; Grafana can need >5m to become ready on slower machines
helm install monitoring-stack prometheus-community/kube-prometheus-stack \
    --version 92.10.1 \
    -f "$SCRIPT_DIR/prometheus-values.yaml" \
    --namespace default \
    --wait --timeout 10m

# ── Deploy services ──────────────────────────────────────────────────────

echo ""
echo "=== Deploying Services ==="

kubectl apply -f "$SCRIPT_DIR/echo_service/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/ml_service/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/ml_service/servicemonitor.yaml"

# ── Wait for readiness ───────────────────────────────────────────────────

echo ""
echo "=== Waiting for pods to be ready ==="

kubectl wait --for=condition=ready pod -l app=echo-service --timeout=120s
kubectl wait --for=condition=ready pod -l app=ml-service --timeout=180s

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "=== Cluster Ready ==="
echo "Echo service:  http://localhost:30080"
echo "ML service: http://localhost:30081"
echo "Prometheus:    http://localhost:30090"
echo "Grafana:       http://localhost:30030  (admin / admin)"
echo ""
kubectl get pods
