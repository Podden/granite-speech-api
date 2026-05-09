FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch first so cache survives app changes.
# CUDA 12.1 wheels by default; build with --build-arg TORCH_INDEX=... to override
# (e.g. https://download.pytorch.org/whl/rocm6.0 for ROCm or .../cpu for CPU).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN pip install --index-url ${TORCH_INDEX} \
        torch==2.7.0 torchaudio==2.7.0

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install .

EXPOSE 8000
VOLUME ["/data/hf-cache"]

CMD ["granite-speech-api"]
