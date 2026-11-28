# Rovik in a container — proxy (:8788) + Laya CPU sidecar (:8777).
#
# UNTESTED AS OF WRITING (2026-11-28): authored on macOS, which has no Docker.
# First build/run must be verified on a real Docker host before this ships.
# See docs/docker.md for the full recipe and open questions.
#
# Design notes:
# - CPU PyTorch only: the container routes; it does not need GPU. Keep the
#   local LLM (and its GPU) on the host and point ROVIK_LOCAL_URL at it.
# - `laya-mlx` is Apple-only and NOT installed here; the `laya` CPU/PyTorch
#   backend serves /predict instead. Same checkpoints, same thresholds.
# - The `rovik start/stop/status` lifecycle manager is deliberately unused:
#   it depends on lsof/ps/SIGTERM and pidfiles. The container *is* the
#   process manager.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Torch CPU first (big, rarely-changing layer) — keeps rebuilds cheap.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch>=2.2.0

# Runtime deps: base package + CPU Laya backend + hub for checkpoint loading.
COPY pyproject.toml README.md ./
COPY rovik/ rovik/
COPY serving/ serving/
RUN pip install --no-deps -e . \
    && pip install fastapi uvicorn "pydantic>=2.12,<3" \
        "huggingface-hub>=1.31.0,<2" transformers safetensors numpy

# UNVERIFIED: the CPU Laya inference package (`laya`, per rovik/preflight.py).
# pip install laya   <- confirmed available? If it is local/wheel-only,
# drop the wheel here or COPY the package source before this line.

# Non-root.
RUN useradd --create-home --uid 1000 rovik
USER rovik

# Ports: proxy (8788) and sidecar (8777). Both bind 0.0.0.0 via env in
# docker-compose.yml — a loopback bind inside a container is unreachable.
EXPOSE 8788 8777