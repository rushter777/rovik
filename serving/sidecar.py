"""
Laya MLX sidecar — FastAPI server on 127.0.0.1:8777.

Loads the configured checkpoint once at startup, guards every prediction
with a threading.Lock (single forward pass at a time), periodically sweeps
the MLX Metal buffer pool to bound memory fragmentation, and serves a /health
endpoint plus a /predict POST endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import gc
import os
import sys
import threading
import warnings

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "LAYA_MODEL_PATH",
    # Production note (author's setup): gate-accuracy parity between two
    # checkpoints was proven head-to-head before swapping (323.1 vs 323.6
    # summed accuracy — a wash), plus a real difficulty head. Point this at
    # whatever checkpoint your own eval says is best.
    "checkpoints/cold-test",
)

# EXACT question text from the training corpus — paraphrased criteria shift the
# probability distribution (measured: paraphrase dropped gate firing 3x).
QUESTION_SET = {
    "tools_file": {
        "type": "noul",
        "instructions": "Does this task involve reading or editing files?",
        "criteria": {"true": "needs file access to do", "false": "no files involved"},
    },
    "tools_terminal": {
        "type": "noul",
        "instructions": "Does this task involve running shell commands or system operations?",
        "criteria": {"true": "needs a terminal or system tools", "false": "no shell needed"},
    },
    "tools_web": {
        "type": "noul",
        "instructions": "Does this task involve looking things up online or fetching web content?",
        "criteria": {"true": "needs web search or fetching", "false": "no web access needed"},
    },
    "needs_verification": {
        "type": "noul",
        "instructions": "Does an answer to this task need double-checking before it is used?",
        "criteria": {"true": "the stakes are high or errors would cost something", "false": "a quick answer is fine and low-stakes"},
    },
    "difficulty": {
        "type": "score",
        "instructions": "Rate how demanding this task is for an assistant, as a count of objective demands: dependent multi-step, discovery/diagnosis, design choice, execution verification.",
        "criteria": ["0-none", "1-one demand", "2-two demands", "3-three demands", "4-four demands"],
    },
}

CLEAR_CACHE_EVERY = int(os.environ.get("LAYA_CLEAR_CACHE_EVERY", "10"))

# ---------------------------------------------------------------------------
# Model loading (once, under lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_agent = None
_load_error: str | None = None
_request_count = 0
_startup_warnings: list[str] = []
_backend_name: str | None = None


def _record_warnings(caught) -> None:
    """Capture unique warning messages from model load/warmup and log them to stderr."""
    for w in caught or ():
        msg = str(getattr(w, "message", w)).strip()
        if not msg:
            continue
        if msg not in _startup_warnings:
            _startup_warnings.append(msg)
        cat = getattr(w, "category", RuntimeWarning)
        cat_name = getattr(cat, "__name__", "Warning")
        print(f"[laya-sidecar] {cat_name}: {msg}", file=sys.stderr, flush=True)


def _clear_mlx_cache():
    """Release MLX's pooled Metal buffers to prevent shape-fragmentation growth."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _import_laya_backend():
    """Import laya_mlx (Apple Silicon MLX) first, falling back to laya (CPU/PyTorch)."""
    try:
        import laya_mlx as laya
        return laya, "laya_mlx"
    except ImportError:
        import laya
        return laya, "laya"


def _load_model():
    """Load the Laya model into global _agent. Must be called under _lock."""
    global _agent, _load_error, _startup_warnings, _backend_name
    _startup_warnings = []
    _backend_name = None
    expanded = os.path.expanduser(MODEL_PATH)
    try:
        laya, backend = _import_laya_backend()
        _backend_name = backend
    except ImportError:
        _agent = None
        _load_error = (
            "Neither 'laya_mlx' (Apple Silicon MLX) nor 'laya' (CPU/PyTorch) is installed. "
            "Install with: pip install laya-mlx (requires huggingface-hub<2)."
        )
        print(f"[laya-sidecar] ERROR: {_load_error}", file=sys.stderr, flush=True)
        return

    if not os.path.exists(expanded):
        _agent = None
        _load_error = (
            f"LAYA_MODEL_PATH does not exist: {expanded}. "
            "Train a checkpoint with 'rovik train' or update LAYA_MODEL_PATH in rovik.env."
        )
        print(f"[laya-sidecar] ERROR: {_load_error}", file=sys.stderr, flush=True)
        return

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            _agent = laya.load(expanded)
        _record_warnings(caught)
        _load_error = None
    except Exception as exc:
        _agent = None
        _load_error = str(exc)
        print(f"[laya-sidecar] ERROR: {_load_error}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up resources on shutdown."""
    global _agent
    with _lock:
        _load_model()
        if _agent is not None:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    _agent.predict("warmup task", QUESTION_SET)
                _record_warnings(caught)
            except Exception:
                pass  # warm-up failure is non-fatal
    try:
        yield
    finally:
        with _lock:
            _agent = None
            _clear_mlx_cache()


app = FastAPI(title="Laya MLX Sidecar", lifespan=lifespan)


@app.get("/health")
def health():
    """Return model load status and captured startup calibration warnings."""
    if _agent is not None:
        return {
            "ok": True,
            "model": MODEL_PATH,
            "backend": _backend_name,
            "requests": _request_count,
            "clear_cache_every": CLEAR_CACHE_EVERY,
            "warnings": list(_startup_warnings),
        }
    return JSONResponse(
        status_code=503,
        content={
            "ok": False,
            "model": MODEL_PATH,
            "error": _load_error or "model failed to load",
            "warnings": list(_startup_warnings),
        },
    )


@app.post("/predict")
def predict(payload: dict):
    """Run a single forward pass through the Laya model."""
    global _request_count
    if _agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not loaded"},
        )

    state = payload.get("state", "")
    questions = payload.get("questions", QUESTION_SET)

    try:
        with _lock:
            result = _agent.predict(state, questions)
            _request_count += 1
            if CLEAR_CACHE_EVERY > 0 and _request_count % CLEAR_CACHE_EVERY == 0:
                _clear_mlx_cache()
        # predict() returns {"answers": {qid: {...}}, "usage": {...}} — surface
        # just the answers map under the same key (no double nesting).
        return {"answers": result.get("answers", result)}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


def main():
    if os.environ.get("ROVIK_SKIP_PREFLIGHT", "").strip().lower() not in ("1", "true", "yes"):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from rovik.preflight import format_preflight, run_preflight

        res = run_preflight(model_path=MODEL_PATH, check_serving=False)
        if not res.ok:
            print("Laya sidecar preflight failed:", file=sys.stderr)
            print(format_preflight(res), file=sys.stderr, flush=True)
            raise SystemExit(1)

    host = os.environ.get("LAYA_SIDECAR_HOST", "127.0.0.1")
    port = int(os.environ.get("LAYA_SIDECAR_PORT", "8777"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
