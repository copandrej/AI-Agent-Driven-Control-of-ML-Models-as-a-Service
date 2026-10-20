FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch + dependencies
RUN pip install --no-cache-dir typing-extensions && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi uvicorn numpy prometheus-client

# Pre-download model weights into the image
COPY download_models.py .
RUN python download_models.py && rm download_models.py

COPY main.py .

# Model version baked into the image: v1 = mobilenet_v2 (default), v2 = resnet50 (see k8s/setup.sh)
ARG MODEL_NAME=mobilenet_v2
ARG MODEL_VERSION=v1
ENV MODEL_NAME=$MODEL_NAME
ENV MODEL_VERSION=$MODEL_VERSION

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
