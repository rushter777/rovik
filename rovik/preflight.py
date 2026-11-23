"""Startup dependency and checkpoint preflight checks for Rovik (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os


@dataclass
class PreflightResult:
    """Structured result of runtime dependency and checkpoint preflight."""

    ok: bool
    laya_backend: str | None
    model_path: str
    model_exists: bool
    missing_runtime: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_laya_backend() -> str | None:
    """Return installed Laya inference backend ('laya_mlx' or 'laya'), else None.

    Uses importlib.util.find_spec so we never initialize Metal or PyTorch during preflight.
    """
    for pkg in ("laya_mlx", "laya"):
        try:
            if importlib.util.find_spec(pkg) is not None:
                return pkg
        except (ImportError, ValueError):
            continue
    return None


def check_serving_runtime() -> list[str]:
    """Return missing core web-serving packages required to run proxy and sidecar."""
    missing: list[str] = []
    for pkg in ("fastapi", "uvicorn"):
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except (ImportError, ValueError):
            missing.append(pkg)
    return missing


def check_model_path(model_path: str) -> tuple[str, bool]:
    """Expand *model_path* and check whether the checkpoint path exists on disk."""
    raw = (model_path or "").strip()
    if not raw:
        return "", False
    expanded = os.path.expanduser(raw)
    return expanded, os.path.exists(expanded)


def run_preflight(
    *,
    model_path: str | None = None,
    check_serving: bool = True,
) -> PreflightResult:
    """Verify Laya inference package, checkpoint path, and serving dependencies."""
    raw_path = model_path if model_path is not None else os.environ.get("LAYA_MODEL_PATH", "checkpoints/cold-test")
    expanded_path, model_exists = check_model_path(raw_path)
    laya_backend = check_laya_backend()
    missing_runtime = check_serving_runtime() if check_serving else []

    errors: list[str] = []
    warnings: list[str] = []

    if laya_backend is None:
        errors.append(
            "Missing Laya inference package: neither 'laya_mlx' (Apple Silicon MLX) nor "
            "'laya' (CPU/PyTorch) is installed. Fix: pip install laya-mlx (requires huggingface-hub<2)."
        )
    if not model_exists:
        shown = expanded_path or repr(raw_path)
        errors.append(
            f"LAYA_MODEL_PATH does not exist: {shown!r}. Fix: run 'rovik train ...' or "
            "point LAYA_MODEL_PATH in rovik.env at an existing checkpoint directory."
        )
    if check_serving and missing_runtime:
        errors.append(
            f"Missing serving dependency: {', '.join(missing_runtime)}. "
            "Fix: pip install -e '.[mlx]' or pip install -r requirements.txt."
        )

    return PreflightResult(
        ok=len(errors) == 0,
        laya_backend=laya_backend,
        model_path=expanded_path,
        model_exists=model_exists,
        missing_runtime=missing_runtime,
        errors=errors,
        warnings=warnings,
    )


def format_preflight(res: PreflightResult) -> str:
    """Format a human-readable summary of preflight checks and actionable fixes."""
    lines: list[str] = []
    if res.laya_backend:
        lines.append(f"  [OK] Laya backend: {res.laya_backend}")
    if res.model_exists:
        lines.append(f"  [OK] Checkpoint path: {res.model_path}")
    if not res.missing_runtime:
        lines.append("  [OK] Serving runtime: fastapi, uvicorn")
    for warn in res.warnings:
        lines.append(f"  [WARN] {warn}")
    for err in res.errors:
        lines.append(f"  [MISSING] {err}")
    return "\n".join(lines)
