FROM python:3.10-slim

ARG MODEL_TYPE=v1

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-learn \
    joblib \
    numpy \
    fastapi \
    uvicorn \
    prometheus-client

COPY train.py .
ENV MODEL_TYPE=${MODEL_TYPE}
RUN python train.py

COPY main.py .

ENV MODEL_VERSION=${MODEL_TYPE}
ENV MODEL_PATH=/app/model.joblib

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
