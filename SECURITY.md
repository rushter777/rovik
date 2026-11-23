# Security Policy

## Supported versions

The `main` branch is the only supported version.

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Instead, use GitHub's private vulnerability reporting on this repo (Security tab → Report a vulnerability), or contact the maintainer directly. Include:

- What you found and its impact
- Steps or a minimal example to reproduce
- Which component: `serving/proxy.py`, `serving/hook.py`, `serving/sidecar.py`, or the pipeline

You'll hear back within a few days. Please avoid public disclosure until a fix is out.

## Security model (so you know what is and isn't in scope)

- Everything is designed to run **on localhost**. The sidecar and proxy bind loopback by default; do not expose them to a network without your own auth in front.
- API keys are read from environment variables at runtime. Never hardcode keys or commit them — including in issues and PRs.
- The judge leg is **fail-open to local by default** (`ROVIK_FALLBACK_ROUTE=local`): if the sidecar fails, prompts route locally without client errors. Escalated frontier upstream failures (e.g. HTTP 401/403 or timeout) **fail loudly** (`502` / SSE error) so auth misconfigurations or demanding tasks are never silently masked or downgraded. API key values are never printed in startup logs, `/health`, or error messages.
- Trained checkpoints and real task corpora are private by policy — they encode private data and are never part of this repo.
