FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch + dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi uvicorn numpy prometheus-client

# Pre-download model weights into the image
COPY download_models.py .
RUN python download_models.py && rm download_models.py

COPY main.py .

# Model selection at runtime (mobilenet_v2 or resnet50)
ENV MODEL_NAME=mobilenet_v2
ENV MODEL_VERSION=v1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
