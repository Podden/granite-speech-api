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
# CUDA 12.1 wheels by default; build with --build-arg TORCH_INDEX=... to override:
#   ROCm:  https://download.pytorch.org/whl/rocm6.0
#   CPU:   https://download.pytorch.org/whl/cpu
#   CUDA:  https://download.pytorch.org/whl/cu121 (default), .../cu124, .../cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN pip install --index-url ${TORCH_INDEX} torch torchaudio

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install .

EXPOSE 8000
VOLUME ["/data/hf-cache"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["granite-speech-api"]
