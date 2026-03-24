"""
MNIST Inference Service
========================
FastAPI service serving a pre-trained digit classifier (scikit-learn).
Uses the sklearn digits dataset (8x8 images, digits 0-9).

Endpoints: /predict, /health, /metrics
Model versioning via MODEL_VERSION env var (matches image tag).
"""

import os
import time

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.joblib")

# ── Prometheus Metrics ────────────────────────────────────────────────────

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "mnist_requests_total",
    "Total inference requests",
    ["model_version", "predicted_digit"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "mnist_request_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

# ── Model Loading ─────────────────────────────────────────────────────────

model = None


def load_model():
    global model
    model = joblib.load(MODEL_PATH)


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(title="MNIST Inference Service", version=MODEL_VERSION)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(request: Request):
    """Accept JSON with 'pixels': list of 64 floats (8x8 image, 0-16 range).
    Returns predicted digit and confidence."""
    start = time.time()

    try:
        body = await request.json()
        pixels = body.get("pixels", [])
        if len(pixels) != 64:
            return {"error": f"Expected 64 pixels (8x8 image), got {len(pixels)}"}
        features = np.array(pixels).reshape(1, -1)
    except Exception:
        return {"error": "Send JSON with 'pixels': list of 64 floats (8x8 image, 0-16 range)"}

    digit = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[digit])
    latency = time.time() - start

    REQUEST_COUNT.labels(model_version=MODEL_VERSION, predicted_digit=str(digit)).inc()
    REQUEST_LATENCY.labels(model_version=MODEL_VERSION).observe(latency)

    return {
        "digit": digit,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency * 1000, 2),
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        generate_latest(registry).decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
