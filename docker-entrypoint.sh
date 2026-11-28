#!/bin/sh
# Rovik container entrypoint — one container, managed lifecycle.
# Sidecar (:8777) + proxy (:8788) run as children of this script via
# `rovik start`; SIGTERM is trapped and forwarded via `rovik stop`.
set -eu
cd /app

# Preflight would probe the local LLM and checkpoint from inside the
# container — boot-order fragile against sibling containers (Ollama may be
# down/restarting). The container is the process manager: skip preflight,
# fail loud at the routing layer instead (ROVIK_FALLBACK_ROUTE=local).
export ROVIK_SKIP_PREFLIGHT=1

# First boot: no checkpoint in the volume yet -> build the documented
# synthetic toy checkpoint (README "1-minute smoke run") so the container
# is self-contained and testable with zero external dependencies.
if [ ! -d "${LAYA_MODEL_PATH:-}" ]; then
  echo '[rovik] No Laya checkpoint found - building the synthetic toy checkpoint.'
  python data/synthetic/generate.py
  python -m rovik train --data data/synthetic/synthetic_corpus.jsonl \
    --output-dir "${LAYA_MODEL_PATH:-/app/checkpoints/cold-test}" \
    --epochs "${ROVIK_TOY_EPOCHS:-2}"
  python -m rovik eval --checkpoint "${LAYA_MODEL_PATH:-/app/checkpoints/cold-test}" \
    --data data/synthetic/synthetic_corpus.jsonl \
    --output /app/data/cold_metrics.json
fi

python -m rovik start --no-wait

stop() {
  python -m rovik stop || true
  exit 0
}
trap stop INT TERM

# Reap children and keep PID 1 alive while the managed services run.
while :; do sleep 3600 & wait $!; done