"""Parameterized agent-harness configuration snippet renderer (stdlib-only)."""

from __future__ import annotations

import json

HARNESS_ORDER = ("hermes", "opencode", "aider", "continue", "generic")

HARNESS_MENU = {
    "1": ("hermes", "Hermes (~/.hermes/profiles/<profile>/config.yaml)"),
    "2": ("opencode", "OpenCode (~/.config/opencode/opencode.json)"),
    "3": ("aider", "Aider (OPENAI_API_BASE + aider --model openai/rovik-proxy)"),
    "4": ("continue", "Continue (~/.continue/config.yaml)"),
    "5": ("generic", "Generic OpenAI Chat Completions (/v1/chat/completions)"),
}


def _render_single(
    name: str,
    *,
    base_url: str,
    local_model: str,
) -> str:
    if name == "hermes":
        return "\n".join(
            [
                "# Hermes (~/.hermes/profiles/<your-agent-profile>/config.yaml)",
                "model:",
                f"  default: {local_model}   # placeholder — proxy overwrites per route",
                "  provider: custom",
                f"  base_url: {base_url}",
                "  api_key: no-key-needed                 # loopback, no auth on proxy",
                "  context_length: 65536",
                "  request_timeout_seconds: 300",
            ]
        )
    if name == "opencode":
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "rovik": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Rovik Router",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "no-key-needed",
                    },
                    "models": {
                        "rovik-proxy": {
                            "name": "Rovik Router",
                        }
                    },
                }
            },
            "model": "rovik/rovik-proxy",
        }
        return "// OpenCode (~/.config/opencode/opencode.json)\n" + json.dumps(cfg, indent=2)
    if name == "aider":
        return "\n".join(
            [
                "# Aider (shell environment + launch)",
                f'export OPENAI_API_BASE="{base_url}"',
                'export OPENAI_API_KEY="no-key-needed"',
                "aider --model openai/rovik-proxy",
            ]
        )
    if name == "continue":
        return "\n".join(
            [
                "# Continue (~/.continue/config.yaml)",
                "name: Rovik Local Router",
                "version: 1.0.0",
                "schema: v1",
                "",
                "models:",
                "  - name: Rovik Router",
                "    provider: openai",
                "    model: rovik-proxy",
                f"    apiBase: {base_url}",
                "    apiKey: no-key-needed",
                "    roles:",
                "      - chat",
                "      - edit",
                "      - apply",
            ]
        )
    if name == "generic":
        return "\n".join(
            [
                "# Generic OpenAI Chat Completions client (/v1/chat/completions)",
                f"base_url = {base_url}",
                f"model    = {local_model}  # or 'rovik-proxy'; overwritten per tier by proxy",
                "api_key  = no-key-needed",
            ]
        )
    raise ValueError(f"Unknown harness {name!r}; expected one of {', '.join(HARNESS_ORDER)} or 'all'")


def render_harness_snippet(
    harness: str = "all",
    *,
    proxy_host: str = "127.0.0.1",
    proxy_port: str | int = "8788",
    local_model: str = "local-35b",
    tag_local: str = "[🟢 LOCAL · 35B-A3B]",
    tag_esc: str = "[⚡ ESCALATED · GLM 5.3]",
) -> str:
    """Render parameterized configuration snippet(s) for supported OpenAI-compatible harnesses."""
    key = (harness or "all").strip().lower()
    host = (proxy_host or "127.0.0.1").strip()
    port = str(proxy_port or "8788").strip()
    model = (local_model or "local-35b").strip()
    base_url = f"http://{host}:{port}/v1"

    if key == "all":
        blocks = [
            f"--- {name} ---\n{_render_single(name, base_url=base_url, local_model=model)}"
            for name in HARNESS_ORDER
        ]
        body = "\n\n".join(blocks)
    elif key in HARNESS_ORDER:
        body = _render_single(key, base_url=base_url, local_model=model)
    else:
        raise ValueError(f"Unknown harness {harness!r}; expected one of {', '.join(HARNESS_ORDER)} or 'all'")

    receipt = f"Expected reply tags: local = {tag_local} | escalated = {tag_esc}"
    return f"{body}\n\n{receipt}"
