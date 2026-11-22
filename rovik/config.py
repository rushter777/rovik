"""Configuration loader and writer for `rovik.env` (stdlib-only)."""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_FILENAME = "rovik.env"

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


def load_config(
    path: str | None = None,
    *,
    override: bool = False,
) -> tuple[str | None, dict[str, str]]:
    """Load config file into os.environ (without overwriting existing env vars unless override=True)."""
    resolved = resolve_config_path(path)
    if not resolved:
        return None, {}
    parsed = parse_env_file(resolved)
    applied: dict[str, str] = {}
    for key, val in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = val
            applied[key] = val
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


def read_secret_value(key_name: str, env_file: str | None = None) -> str:
    """Read an API key from os.environ first, falling back to env_file (never printed)."""
    if not key_name:
        return ""
    val = os.environ.get(key_name, "").strip()
    if val:
        return val
    ef = env_file if env_file is not None else os.environ.get("ROVIK_ENV_FILE", "~/.hermes/.env")
    if not ef:
        return ""
    expanded = os.path.expanduser(ef)
    if not os.path.isfile(expanded):
        return ""
    return parse_env_file(expanded).get(key_name, "").strip()
