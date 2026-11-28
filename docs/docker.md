# Rovik in Docker

**Status: authored 2026-11-28, UNTESTED** — written on a Docker-less mac (Docker
was deliberately purged from that machine). The first person to `docker compose
up` on a real host owns the verification pass; until then, treat every command
below as a best-evidence draft, not a receipt.

## Why Docker, and why ONE container

The base package (`fastapi`, `uvicorn`, `pydantic`) is pure Python, so a
container makes Rovik run anywhere Docker runs — including a Windows laptop
with zero code changes to the routing layer. The `rovik start/stop/status`
lifecycle manager (lsof/ps/SIGTERM pidfiles) also runs unmodified inside
Linux, so we use it: **one container runs the whole managed lifecycle**, with
sidecar (:8777) and proxy (:8788) as children of `docker-entrypoint.sh`.

Single-container is deliberate:

- the repo's **loopback defaults become correct** — the judge URL
  (`http://127.0.0.1:8777/predict`) just works, no cross-container networking;
- no `depends_on`, no service-name DNS for the proxy→sidecar hop;
- only the proxy is published; the sidecar never leaves container loopback.

## Shape

```
docker compose up -d          # builds rovik:latest

rovik (one container)
  ├─ python -m rovik sidecar   (Laya CPU backend, 127.0.0.1:8777)
  └─ python -m rovik proxy     (0.0.0.0:8788, published)
```

Agent harnesses on the same docker network use `http://rovik:8788/v1`.
Harness containers *outside* this compose file need to join the network
(`ROVIK_NETWORK`) or use the published host port.

## Non-negotiables (the traps)

1. **`ROVIK_PROXY_HOST=0.0.0.0`** — a loopback bind inside a container is
   unreachable from every other container and from the host. Symptom: instant
   `connection refused`. The **sidecar** stays on `127.0.0.1`: only the proxy
   (same container) talks to it.
2. **`ROVIK_SKIP_PREFLIGHT=1`** — `rovik start` preflights the local LLM and
   checkpoint, which is boot-order fragile against sibling containers (Ollama
   may still be restarting). Skip it; `ROVIK_FALLBACK_ROUTE=local` makes
   routing fail loud and free instead.
3. **CPU torch before `.[train]`** — `pip install --index-url
   https://download.pytorch.org/whl/cpu torch` FIRST, or pip resolves the CUDA
   build and the image balloons by gigabytes the CPU can never use.
4. **`laya` (CPU backend), never `.[mlx]` off Apple Silicon** — MLX does not
   build on Linux/Windows. `laya>=0.5.5` is verified on PyPI (0.5.6, Apache-2.0,
   Python ≥3.10).
5. **`lsof` + `procps` in the image** — python:3.11-slim lacks them and rovik
   degrades silently: `stop` still works (real pids, SIGTERM) but port-conflict
   detection and `--force-port` are dead. `apt-get install` lights them up.
6. **`ROVIK_FALLBACK_ROUTE=local`** (default, set it explicitly anyway) — Laya
   or local-model failure must never escalate to the paid tier.
7. **Lower `LAYA_THRESH_HARD` for small local models** — 0.5–0.6 for a 7B,
   per the main README §Three Model Roles; 0.8 is for a 35B.
8. **Secrets stay out of the image** — keys arrive via runtime env /
   `env_file` (`rovik.docker.env`, gitignored, `required: false`); nothing is
   baked into a layer, no key is ever committed.
9. **Copy `pipeline/`** — `rovik train/eval/sweep` imports it at runtime.

## First boot

`docker-entrypoint.sh` checks `LAYA_MODEL_PATH`; if the mounted volume has no
checkpoint yet, it builds the documented synthetic toy checkpoint (README
"1-minute smoke run": `generate.py` → train → eval) so the container is fully
self-contained. Bring your own trained checkpoint by dropping it into the
`rovik-checkpoints` volume and pointing `LAYA_MODEL_PATH` at it.

## Open questions (verify on real hardware)

- **Checkpoint format under the CPU backend.** The `laya` PyPI package loads
  classic-encoder checkpoints from the HF Hub (`convaiinnovations/laya`).
  Whether a local `LAYA_MODEL_PATH` checkpoint directory (as produced by
  `pipeline/train_rlcd.py` / the MLX path) loads as-is under the CPU backend
  — or needs an export/Hub-id step — is unverified. First boot's synthetic
  train answers it empirically: if that works, your own checkpoint in the same
  format works.
- **`rovik start --no-wait`** — the flag exists in `start_services(wait=False)`;
  confirm the CLI spelling on the first real run.
- Preflight skip also skips checkpoint sanity checks — run
  `python -m rovik config` inside the container if you want the look.

## Verified findings (real CPU-container build, 2026-11-28)

A third-party deployment on a Windows laptop (i7-13700 / RTX 3060, Docker
Desktop) exercised the first real build of this recipe. Three fixes landed
from that run — treat these as build-verified, unlike the rest of this draft:

1. **`.dockerignore` must NOT exclude `data/`** — the entrypoint needs
   `data/synthetic/` for the first-boot toy checkpoint, and `COPY data ./data`
   fails the build if the directory is ignored. Only live artifacts
   (`data/*.log`, `data/*.pid`, `data/proxy_log.jsonl`, `data/*.json`) stay out.
2. **`pipeline/train_rlcd.py` called `torch.mps.empty_cache()` unguarded** —
   a hard crash on any CPU-only backend ("Cannot execute emptyCache() without
   MPS backend"). Now guarded behind `torch.backends.mps.is_available()`.
   The toy-smoke-run in a CPU container doubles as the cross-platform
   regression test for this.
3. **`/app/checkpoints` must exist and be `rovik`-owned in the image** —
   Docker initializes an empty named volume with the image mount point's
   ownership; without the `mkdir + chown` the volume mounts root-owned and
   first-boot training dies with `PermissionError`, restart-looping the
   container.

Also confirmed environmental, no code change: host port 8788 may conflict
(`ROVIK_PORT=18788` worked), and a stale same-name container blocks
`docker compose up -d` until removed.

## Smoke test (once on real hardware)

```bash
docker compose build
docker compose up -d
docker compose exec rovik python -m rovik status || true
curl -s http://localhost:8788/ | head        # dashboard
curl -s http://localhost:8788/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"rovik","messages":[{"role":"user","content":"hi"}]}'
docker compose logs rovik | tail -50
```

Expect the LOCAL tag on trivial prompts and no frontier traffic unless a hard
prompt crosses `LAYA_THRESH_HARD`. The dashboard at `/` shows route mix,
savings, and per-decision p_hard.

## Windows notes

Docker Desktop on Windows runs all of this on WSL2 with no code changes.
There is nothing Mac-specific in the container: CPU torch, `laya` CPU backend,
pure-Python proxy. A local LLM on the Windows host is reached with
`ROVIK_LOCAL_URL=http://host.docker.internal:<port>/...` (Docker Desktop
defines that name natively); a GPU (6GB-class) stays with the host LLM, never
inside this container.