# Rovik in a container — ONE container running the managed lifecycle:
# proxy (:8788) + Laya CPU sidecar (:8777) as children of docker-entrypoint.sh.
#
# UNTESTED AS OF WRITING (2026-11-28): authored on macOS, which has no Docker
# (deliberately purged). First build/run must be verified on a real Docker
# host. See docs/docker.md.
#
# Design (single container — see docs/docker.md and rovik-router-ops):
# - Both services in ONE container => the repo's loopback defaults become
#   CORRECT (judge URL http://127.0.0.1:8777, no cross-container networking).
# - `rovik start/stop` runs unmodified; the entrypoint traps SIGTERM and
#   forwards it. `lsof`/`ps` are installed so rovik's port-conflict
#   detection and `--force-port` actually work (python:3.11-slim lacks them
#   and rovik degrades silently without).
# - CPU-only torch FIRST so `.[train]`'s torch>=2.2 can't pull CUDA builds
#   (multi-GB of wheels this CPU container can never use).
# - `laya` PyPI package = CPU backend (v0.5.6 verified on PyPI, Apache-2.0,
#   py>=3.10). `.[mlx]` is Apple-only and must NOT appear here.
# - `pipeline/` is copied: `rovik train/eval/sweep` imports it at runtime.
# - No secrets in any layer: keys arrive via runtime env / env_file only.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4

# lsof/procps: rovik lifecycle port-conflict detection + --force-port.
RUN apt-get update \
 && apt-get install -y --no-install-recommends lsof procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch BEFORE anything that requires torch>=2.2.
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY rovik ./rovik
COPY serving ./serving
COPY pipeline ./pipeline
COPY data ./data

RUN python -m pip install '.[train]' \
 && python -m pip install 'laya>=0.5.5'

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Non-root: a routing proxy reachable from other containers should not own uid 0.
# /app/checkpoints must EXIST and be owned by rovik BEFORE the named volume
# mounts over it: Docker initializes an empty named volume with the image
# mount point's ownership, so without this the volume comes up root-owned and
# first-boot checkpoint training dies with PermissionError (found in a real
# CPU-container build, 2026-11-28).
RUN useradd --create-home --uid 1000 rovik \
 && mkdir -p /app/checkpoints /app/data \
 && chown -R rovik:rovik /app
USER rovik

# Only the proxy is published; the sidecar stays on container loopback.
EXPOSE 8788

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]