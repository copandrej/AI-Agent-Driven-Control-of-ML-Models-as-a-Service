"""
ML Inference Service
====================
FastAPI service serving pre-trained ImageNet classifiers (PyTorch/torchvision).
No training required — models are downloaded at Docker build time.

MODEL_NAME env var selects the architecture:
  mobilenet_v2  (v1 — lighter, ~10-30ms inference on CPU)
  resnet50      (v2 — heavier, ~30-80ms inference on CPU)

Endpoints: /predict, /health, /metrics
"""

import os
import time

import numpy as np
import torch
import torchvision.models as models
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

AVAILABLE_MODELS = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

# ── Prometheus Metrics ────────────────────────────────────────────────────

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "ml_requests_total",
    "Total inference requests",
    ["model_version", "predicted_class"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "ml_request_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=registry,
)

# ── Model Loading ─────────────────────────────────────────────────────────

model = None
categories = None


def load_model():
    global model, categories
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {MODEL_NAME}. Available: {list(AVAILABLE_MODELS)}"
        )
    model_fn, weights = AVAILABLE_MODELS[MODEL_NAME]
    model = model_fn(weights=weights)
    model.eval()
    categories = weights.meta["categories"]


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(title="ML Inference Service", version=MODEL_VERSION)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_version": MODEL_VERSION,
        "model_name": MODEL_NAME,
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(request: Request):
    """Accept JSON with optional 'data' field (1x3x224x224 tensor).
    If omitted, generates random input — useful for load testing.
    Returns top-5 ImageNet predictions with confidence scores."""
    start = time.time()

    try:
        body = await request.json()
    except Exception:
        body = {}

    if "data" in body:
        try:
            data = torch.tensor(body["data"], dtype=torch.float32)
            if data.dim() == 1:
                data = data.reshape(1, 3, 224, 224)
            elif data.dim() == 3:
                data = data.unsqueeze(0)
        except Exception:
            return {"error": "Invalid data shape. Expected 1x3x224x224 tensor."}
    else:
        data = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(data)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_idx = torch.topk(probabilities, 5)
    top5 = [
        {"class": categories[idx], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top5_prob, top5_idx)
    ]

    latency = time.time() - start
    predicted_class = top5[0]["class"]

    REQUEST_COUNT.labels(
        model_version=MODEL_VERSION, predicted_class=predicted_class
    ).inc()
    REQUEST_LATENCY.labels(model_version=MODEL_VERSION).observe(latency)

    return {
        "predictions": top5,
        "latency_ms": round(latency * 1000, 2),
        "model_version": MODEL_VERSION,
        "model_name": MODEL_NAME,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        generate_latest(registry).decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
