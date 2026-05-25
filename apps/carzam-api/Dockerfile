# --------- carzam-api container ---------
# Build context = monorepo root so we can pull in src/carzam (the model code).
# At repo root run: docker build -f apps/carzam-api/Dockerfile -t carzam-api .

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by soundfile + librosa (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install CPU-only PyTorch BEFORE carai so pip doesn't auto-resolve the
# default torch wheel. As of torch 2.6+, the default Linux x86_64 wheel
# bundles 2+ GB of CUDA 13 libraries (cublas, cudnn, nccl, triton, ...) —
# useless on a CPU-only Hetzner box and enough to fill /var on the build
# host. The CPU index serves a slim ~150 MB wheel with no CUDA deps.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchaudio==2.5.1

# Install carai (the model lib). torch is already satisfied so pip won't
# pull a different version from PyPI.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Install the API on top.
COPY apps/carzam-api/pyproject.toml apps/carzam-api/
COPY apps/carzam-api/app apps/carzam-api/app
RUN pip install --no-cache-dir -e ./apps/carzam-api

# The model checkpoint (~320MB) lives in R2, not the image — fetched on
# container start. Keeps the image ~1GB instead of ~1.4GB and lets you swap
# checkpoints by re-uploading to R2 + restarting the container.
RUN mkdir -p /data
ENV CHECKPOINT_PATH=/data/checkpoint.pt \
    CLASSES_PATH=/data/classes.json

EXPOSE 8000
# start-period accounts for: 318MB checkpoint download from R2 (~10-30s) +
# CNN14 weights load into RAM (~20-40s) + Postgres init + seed (~1-2s).
# 180s gives headroom even on a cold CX21 with concurrent old/new containers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
