FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/data/hf-cache

# uv for fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# gcc: triton JIT-compiles its CUDA driver stub at runtime (torch native ops
# route e.g. RoPE matmuls through triton kernels) and needs a C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA 12.8 wheels by default (first version with torch ≥ 2.7).
# Override with --build-arg TORCH_INDEX=... :
#   ROCm:  https://download.pytorch.org/whl/rocm6.3
#   CPU:   https://download.pytorch.org/whl/cpu
#   CUDA:  https://download.pytorch.org/whl/cu121, cu124, cu128 (default)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# Install torch first — layer cached independently of app code changes.
# TORCH_INDEX is primary (CUDA wheels); PyPI is extra (provides torch deps like filelock).
RUN uv pip install --system \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url "https://pypi.org/simple" \
        "torch>=2.7" "torchaudio>=2.7"

COPY pyproject.toml README.md ./
COPY app ./app

# Install remaining app deps.
# --extra-index-url lets uv find the already-installed CUDA torch (>=2.7 satisfied)
# so it is NOT replaced by a PyPI rebuild. unsafe-best-match: the torch index
# mirrors stale copies of common packages (e.g. tqdm<=4.66.5) and uv's default
# first-index strategy would pin those, breaking pyannote's resolution.
RUN uv pip install --system \
        --index-strategy unsafe-best-match \
        --extra-index-url "${TORCH_INDEX}" \
        .

EXPOSE 8000
VOLUME ["/data/hf-cache"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["granite-speech-api"]
