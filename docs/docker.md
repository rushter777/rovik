# Rovik in Docker

**Status: authored 2026-11-28, UNTESTED** — written on a Docker-less mac (Docker
was deliberately purged from that machine). The first person to `docker compose
up` on a real host owns the verification pass; until then, treat every command
below as a best-evidence draft, not a receipt.

## Why Docker

The container replaces `rovik start/stop/status` — the lifecycle manager depends
on `lsof`, `ps`, and SIGTERM pidfiles and is macOS/WSL-shaped. Inside a container
you don't need it: the container *is* the process manager. The base package
(`fastapi`, `uvicorn`, `pydantic`) is pure Python, so containerizing makes Rovik
run anywhere Docker runs — including a friend's Windows laptop with zero code
changes to the routing layer.

## Shape

```
docker compose up -d          # builds rovik:latest, starts sidecar + proxy

rovik-sidecar   :  python -m rovik sidecar   (Laya CPU, :8777, 0.0.0.0)
rovik           :  python -m rovik proxy     (:8788, 0.0.0.0, published)
```

Agent harnesses on the same compose network use `http://rovik:8788/v1`.
Harness containers *outside* this compose file need to join the network or use
the published host port.

## Non-negotiables (the traps)

1. **`ROVIK_PROXY_HOST=0.0.0.0` and `LAYA_SIDECAR_HOST=0.0.0.0`.** The defaults
   are loopback; a loopback bind inside a container is unreachable from any
   other container. Symptom: instant `connection refused`.
2. **`ROVIK_JUDGE_URL=http://rovik-sidecar:8777/predict`** — the proxy must reach
   the sidecar by service name, not 127.0.0.1, for exactly the same reason.
3. **The local LLM stays on the host** (keep the GPU out of the container).
   `ROVIK_LOCAL_URL=http://host.docker.internal:8081/...` bridges back out.
   On the author's Mac the local model is llama-server on :8081; on Windows
   the same shape holds for llama.cpp / Ollama / LM Studio.
4. **`ROVIK_FALLBACK_ROUTE=local` and lower `LAYA_THRESH_HARD`** for small
   local models (0.5–0.6, per the main README §Three Model Roles).
5. **Secrets stay out of the image.** API keys come via env / env_file at
   runtime; nothing is baked into a layer. Do not commit a `rovik.env` with a key.

## Open questions (verify before trusting)

- **CPU Laya package availability.** `rovik/preflight.py` names two backends:
  `laya_mlx` (Apple) and `laya` (CPU/PyTorch). The Dockerfile assumes `pip install laya`
  works. If `laya` is not published (wheel/local only), fix the Dockerfile before
  the first build.
- **Checkpoint mount.** `LAYA_MODEL_PATH=/checkpoints/laya` assumes the MLX
  checkpoint directory loads as-is under the CPU backend. If the CPU backend
  needs a different export format, the compose volume/mount changes — verify.
- **Preflight is skipped** (`ROVIK_SKIP_PREFLIGHT=1`) because it probes things
  the container can't see. That also skips checkpoint sanity checks — run
  `python -m rovik config` once inside the container if you want a look.

## Smoke test (once on real hardware)

```bash
docker compose build
docker compose up -d
docker compose exec rovik python -m rovik status || true
curl -s http://localhost:8788/ | head        # dashboard
curl -s http://localhost:8788/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"rovik","messages":[{"role":"user","content":"hi"}]}'
docker compose logs rovik rovik-sidecar | tail -50
```

Expect the LOCAL tag on trivial prompts and no frontier traffic unless a
hard prompt crosses `LAYA_THRESH_HARD`.