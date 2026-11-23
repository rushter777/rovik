"""Configuration loader and writer for `rovik.env` (stdlib-only)."""

from __future__ import annotations

import os
from typing import Mapping

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_FILENAME = "rovik.env"

LOADED_CONFIG_PATH: str | None = None
ENV_OVERRIDES: dict[str, tuple[str, str]] = {}

_SECRET_SUFFIXES = ("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD")

KNOWN_KEYS = (
    "ROVIK_LOCAL_URL",
    "ROVIK_LOCAL_MODEL",
    "ROVIK_TAG_LOCAL",
    "ROVIK_ESC_URL",
    "ROVIK_ESC_MODEL",
    "ROVIK_ESC_KEYNAME",
    "ROVIK_ENV_FILE",
    "ROVIK_FRONTIER_PRICE_PER_MTOK",
    "ROVIK_TAG_ESC",
    "ROVIK_TEACHER_URL",
    "ROVIK_TEACHER_MODEL",
    "ROVIK_TEACHER_KEYNAME",
    "ROVIK_CORPUS_PATH",
    "LAYA_MODEL_PATH",
    "LAYA_SIDECAR_HOST",
    "LAYA_SIDECAR_PORT",
    "LAYA_SIDECAR_URL",
    "LAYA_CLEAR_CACHE_EVERY",
    "LAYA_THRESH_HARD",
    "LAYA_THRESH_TOOLS",
    "LAYA_THRESH_VERIFY",
    "ROVIK_PROXY_HOST",
    "ROVIK_PROXY_PORT",
    "ROVIK_JUDGE_URL",
    "ROVIK_FALLBACK_ROUTE",
    "ROVIK_LOG_DIR",
)


def _unquote_env_value(raw: str) -> str:
    """Strip matching outer quotes and unescape double-quoted escape sequences."""
    val = raw.strip()
    if len(val) >= 2 and val[0] == val[-1] == "'":
        return val[1:-1]
    if len(val) >= 2 and val[0] == val[-1] == '"':
        inner = val[1:-1]
        out: list[str] = []
        i = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "\\" and i + 1 < len(inner) and inner[i + 1] in ('"', "\\", "$"):
                out.append(inner[i + 1])
                i += 2
            else:
                out.append(ch)
                i += 1
        return "".join(out)
    return val


def parse_env_file(path: str) -> dict[str, str]:
    """Parse KEY=VALUE pairs from an env file, ignoring comments and blanks."""
    values: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("export "):
                raw = raw[len("export "):].lstrip()
            if "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _unquote_env_value(val)
    return values


def resolve_config_path(path: str | None = None) -> str | None:
    """Locate an existing config file by explicit path, ROVIK_CONFIG, cwd, or repo root."""
    if path:
        expanded = os.path.expanduser(path)
        return expanded if os.path.isfile(expanded) else None

    env_cfg = os.environ.get("ROVIK_CONFIG", "").strip()
    if env_cfg:
        expanded = os.path.expanduser(env_cfg)
        return expanded if os.path.isfile(expanded) else None

    candidates = (
        os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME),
        os.path.join(REPO_ROOT, DEFAULT_CONFIG_FILENAME),
    )
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def is_secret_key(key: str) -> bool:
    """Return True if *key* names a credential whose value must never be printed."""
    return any(key.endswith(sfx) for sfx in _SECRET_SUFFIXES)


def load_config(
    path: str | None = None,
    *,
    override: bool = False,
) -> tuple[str | None, dict[str, str]]:
    """Load config file into os.environ (without overwriting existing env vars unless override=True)."""
    global LOADED_CONFIG_PATH, ENV_OVERRIDES
    resolved = resolve_config_path(path)
    LOADED_CONFIG_PATH = resolved
    ENV_OVERRIDES = {}
    if not resolved:
        return None, {}
    parsed = parse_env_file(resolved)
    applied: dict[str, str] = {}
    for key, val in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = val
            applied[key] = val
        elif os.environ[key] != val:
            ENV_OVERRIDES[key] = (os.environ[key], val)
    return resolved, applied


def format_env_value(value: str) -> str:
    """Format an env value safely, quoting when whitespace or special shell chars exist."""
    s = str(value)
    if s == "":
        return ""
    needs_quote = any(c.isspace() or c in ('#', "'", '"', "$", "\\") for c in s)
    if not needs_quote:
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def write_env_file(
    path: str,
    values: dict[str, str],
    header_comments: list[str] | None = None,
) -> str:
    """Write KEY=VALUE pairs to disk with 0o600 permissions."""
    expanded = os.path.expanduser(path)
    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines: list[str] = []
    if header_comments:
        for comment in header_comments:
            c = comment if comment.startswith("#") else f"# {comment}"
            lines.append(c)
        lines.append("")
    for key, val in values.items():
        lines.append(f"{key}={format_env_value(val)}")
    with open(expanded, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(expanded, 0o600)
    except OSError:
        pass
    return expanded


def detect_env_overrides(
    config_path: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[dict]:
    """Return config entries shadowed by differing values in *env* (never exposes secret values)."""
    resolved = resolve_config_path(config_path) if config_path else (LOADED_CONFIG_PATH or resolve_config_path())
    if not resolved:
        return []
    parsed = parse_env_file(resolved)
    target_env = os.environ if env is None else env
    findings: list[dict] = []
    for key, file_val in parsed.items():
        if key not in target_env:
            continue
        env_val = target_env[key]
        if env_val == file_val:
            continue
        secret = is_secret_key(key)
        if secret:
            warning = f"{key} in process environment overrides a different value in {resolved} (values redacted)."
        else:
            warning = f"{key} in process environment ({env_val!r}) overrides {resolved} ({file_val!r})"
        findings.append(
            {
                "key": key,
                "config_path": resolved,
                "env_value": env_val,
                "file_value": file_val,
                "is_secret": secret,
                "warning": warning,
            }
        )
    return findings


def diagnose_key_source(
    key_name: str,
    env_file: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Resolve an API key's source ('env' | 'env-file' | 'missing') and detect stale conflicts.

    Returned warnings never include any secret value.
    """
    target_env = os.environ if env is None else env
    name = (key_name or "").strip()
    env_val = (target_env.get(name, "") or "").strip() if name else ""

    ef = env_file if env_file is not None else target_env.get("ROVIK_ENV_FILE", "~/.hermes/.env")
    expanded = os.path.expanduser(ef) if ef else ""
    file_val = ""
    if name and expanded and os.path.isfile(expanded):
        file_val = parse_env_file(expanded).get(name, "").strip()

    if env_val:
        source = "env"
        value = env_val
    elif file_val:
        source = "env-file"
        value = file_val
    else:
        source = "missing"
        value = ""

    conflict = bool(env_val and file_val and env_val != file_val)
    warning: str | None = None
    if conflict:
        warning = (
            f"{name} in shell environment overrides a different value in {expanded} "
            f"(run 'unset {name}' to use {expanded})."
        )
    elif source == "missing" and name:
        target_desc = expanded or "(no ROVIK_ENV_FILE)"
        warning = (
            f"{name} is not set in os.environ or {target_desc}; "
            "escalated requests will likely fail auth."
        )

    return {
        "key_name": name,
        "value": value,
        "source": source,
        "env_file": expanded,
        "env_present": bool(env_val),
        "env_file_present": bool(file_val),
        "conflict": conflict,
        "warning": warning,
    }


def read_secret_value(key_name: str, env_file: str | None = None) -> str:
    """Read an API key from os.environ first, falling back to env_file (never printed)."""
    return diagnose_key_source(key_name, env_file)["value"]
