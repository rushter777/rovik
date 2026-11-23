#!/usr/bin/env python3
"""Rovik proxy — the local-first model routing switch.

OpenAI-compatible /v1/chat/completions endpoint that routes per request:

    request comes in
      └─ judge latest user task with Laya (sidecar :8777)
            ├─ difficulty flag (p_hard >= 0.8) → ESCALATE / passthrough-harness
            │   to the session model (GLM 5.3 via OpenRouter)
            └─ routine (including simple tool calls) → LOCAL / local-harness:
                the 35B MoE answers (llama-server :8081), free

Every answer is tagged so the human always knows who answered:
    [🟢 LOCAL · 35B-A3B]   or   [⚡ ESCALATED · GLM 5.3]
Every decision is logged to data/proxy_log.jsonl (ROVIK_LOG_DIR to override).

Fail-safe matrix:
    - Judge unavailable / timeout -> ROVIK_FALLBACK_ROUTE (default 'local', or 'escalate').
    - Local upstream error          -> escalate to frontier (rule='local-failed-escalate').
    - Frontier upstream error       -> deliberate FAIL-LOUD (HTTP 502 / SSE error event
      with upstream status, body, and non-secret key-source diagnostics; never silently
      downgrades an escalated request to local or masks auth misconfigurations).

Env:
    ROVIK_PROXY_HOST            (default 127.0.0.1)
    ROVIK_PROXY_PORT            (default 8788)
    ROVIK_LOCAL_URL             (default http://127.0.0.1:8081/v1/chat/completions)
    ROVIK_LOCAL_MODEL           (default local-35b)
    ROVIK_JUDGE_URL             (default http://127.0.0.1:8777/predict; falls back to LAYA_SIDECAR_URL)
    ROVIK_ESC_URL               (default https://openrouter.ai/api/v1/chat/completions)
    ROVIK_ESC_MODEL             (default z-ai/glm-5.3 — swap frontier providers without code changes)
    ROVIK_ESC_KEYNAME           (default OPENROUTER_API_KEY — env var holding the key)
    ROVIK_ENV_FILE              (default ~/.hermes/.env — fallback env file if key not in os.environ)
    ROVIK_FALLBACK_ROUTE        (default local — 'local' or 'escalate' when judge fails)
    ROVIK_LOG_DIR               (default <repo>/data)
    ROVIK_TAG_LOCAL             (default '[🟢 LOCAL · 35B-A3B]')
    ROVIK_TAG_ESC               (default '[⚡ ESCALATED · GLM 5.3]')
    ROVIK_FRONTIER_PRICE_PER_MTOK (default 3.0)
    LAYA_THRESH_HARD            (default 0.8)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rovik import config as rovik_config  # noqa: E402

try:
    from serving import hook as laya_hook
except ImportError:
    sys.path.insert(0, _HERE)
    import hook as laya_hook

PROXY_PORT = int(os.environ.get("ROVIK_PROXY_PORT", "8788"))
PROXY_HOST = os.environ.get("ROVIK_PROXY_HOST", "127.0.0.1")
LOCAL_URL = os.environ.get("ROVIK_LOCAL_URL", "http://127.0.0.1:8081/v1/chat/completions")
LOCAL_MODEL = os.environ.get("ROVIK_LOCAL_MODEL", "local-35b")
JUDGE_URL = os.environ.get("ROVIK_JUDGE_URL") or os.environ.get(
    "LAYA_SIDECAR_URL", "http://127.0.0.1:8777/predict"
)
ESC_KEYNAME = os.environ.get("ROVIK_ESC_KEYNAME", "OPENROUTER_API_KEY")
ENV_FILE = os.environ.get("ROVIK_ENV_FILE", "~/.hermes/.env")
# Frontier endpoint — swap providers without touching code (OpenRouter by default).
ESC_URL = os.environ.get(
    "ROVIK_ESC_URL", "https://openrouter.ai/api/v1/chat/completions"
)
THRESH = float(os.environ.get("LAYA_THRESH_HARD", "0.8"))
FALLBACK_ROUTE = os.environ.get(
    "ROVIK_FALLBACK_ROUTE", os.environ.get("FALLBACK_ROUTE", "local")
).strip().lower()
if FALLBACK_ROUTE not in ("local", "escalate"):
    FALLBACK_ROUTE = "local"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LOG_DIR = os.environ.get("ROVIK_LOG_DIR", os.path.join(REPO_ROOT, "data"))
LOG_FILE = os.path.join(LOG_DIR, "proxy_log.jsonl")
# Tags are user-facing receipts, fully configurable. Defaults describe THIS
# machine's stack; override per deployment (icons-only works too, e.g.
# ROVIK_TAG_LOCAL="🟢"). History-stripping recognizes any bracketed
# 🟢/⚡ tag regardless of exact text, so old tags never poison context.
TAG_LOCAL = os.environ.get("ROVIK_TAG_LOCAL", "[🟢 LOCAL · 35B-A3B]")
TAG_ESC = os.environ.get("ROVIK_TAG_ESC", "[⚡ ESCALATED · GLM 5.3]")

ESC_FAILURE_POLICY = "fail-loud"


class UpstreamHTTPError(RuntimeError):
    """Upstream HTTP error carrying status code and sanitized upstream error body."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def _read_http_error_body(exc: urllib.error.HTTPError, limit: int = 500) -> str:
    """Safely extract and compact an HTTPError response body up to *limit* chars."""
    try:
        raw = exc.read()
    except Exception:
        return ""
    if not raw:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", "replace")
    else:
        text = str(raw)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _auth_key_hint(key_info: dict | None) -> str:
    """Return non-secret key-source context for 401/403 escalation errors."""
    if not key_info:
        return ""
    key_name = key_info.get("key_name") or ESC_KEYNAME
    if key_info.get("conflict"):
        return f" [key {key_name} loaded from env; differs from {key_info.get('env_file')}]"
    return f" [key {key_name} source: {key_info.get('source', 'unknown')}]"


def _format_http_error(
    exc: urllib.error.HTTPError,
    *,
    key_info: dict | None = None,
    body: str | None = None,
) -> str:
    """Format HTTP status, reason, upstream response body, and auth key source."""
    err_body = _read_http_error_body(exc) if body is None else body
    msg = f"HTTP Error {exc.code}: {exc.reason}"
    if err_body:
        msg = f"{msg} — {err_body}"
    if exc.code in (401, 403) and key_info:
        msg = f"{msg}{_auth_key_hint(key_info)}"
    return msg


def _esc_key_info() -> dict:
    """Resolve frontier key source using current module globals ESC_KEYNAME / ENV_FILE."""
    return rovik_config.diagnose_key_source(ESC_KEYNAME, ENV_FILE)


def _esc_headers():
    """Read ESC_KEYNAME from os.environ first, falling back to ROVIK_ENV_FILE. Never printed."""
    headers = {"Content-Type": "application/json"}
    api_key = _esc_key_info()["value"]
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _collect_startup_diagnostics() -> tuple[dict, list[str]]:
    """Summarize frontier key source (never secret value) and active env override warnings."""
    info = _esc_key_info()
    key_summary = {
        "name": info["key_name"],
        "source": info["source"],
        "env_file": info["env_file"],
        "conflict": info["conflict"],
    }
    warnings_list: list[str] = []
    if info["warning"]:
        warnings_list.append(info["warning"])

    seen_keys: set[str] = set()
    for item in rovik_config.detect_env_overrides():
        k = item["key"]
        if not (k.startswith("ROVIK_") or k.startswith("LAYA_")):
            continue
        seen_keys.add(k)
        warnings_list.append(item["warning"])

    if rovik_config.LOADED_CONFIG_PATH:
        for k, (env_val, file_val) in rovik_config.ENV_OVERRIDES.items():
            if k in seen_keys or not (k.startswith("ROVIK_") or k.startswith("LAYA_")):
                continue
            if rovik_config.is_secret_key(k):
                warnings_list.append(
                    f"{k} in process environment overrides a different value in "
                    f"{rovik_config.LOADED_CONFIG_PATH} (values redacted)."
                )
            else:
                warnings_list.append(
                    f"{k} in process environment ({env_val!r}) overrides "
                    f"{rovik_config.LOADED_CONFIG_PATH} ({file_val!r})"
                )
    return key_summary, warnings_list


def log_startup_diagnostics(stream=None) -> list[str]:
    """Log frontier key source and env-override warnings at startup (never prints key values)."""
    out = sys.stderr if stream is None else stream
    key_summary, warnings_list = _collect_startup_diagnostics()
    src_detail = key_summary["source"]
    if src_detail == "env-file" and key_summary["env_file"]:
        src_detail = f"env-file ({key_summary['env_file']})"
    print(f"[rovik-proxy] Frontier key {key_summary['name']}: source={src_detail}", file=out)
    for warn in warnings_list:
        print(f"[rovik-proxy] WARNING: {warn}", file=out)
    if hasattr(out, "flush"):
        out.flush()
    return warnings_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Emit key-source and environment override diagnostics on proxy startup."""
    log_startup_diagnostics()
    yield


app = FastAPI(title="Rovik Proxy", lifespan=lifespan)


def _post(url, payload, headers, timeout_s=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            body = _read_http_error_body(exc)
            msg = _format_http_error(exc, body=body)
            code = exc.code
        finally:
            try:
                exc.close()
            except Exception:
                pass
        raise UpstreamHTTPError(code, msg, body=body) from exc


def _latest_user_text(messages):
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):  # multimodal: take text parts
                txt = " ".join(p.get("text", "") for p in c
                               if isinstance(p, dict) and p.get("type") == "text")
                if txt.strip():
                    return txt
    return ""


def _strip_tags_from_history(payload):
    """Remove Rovik tags from assistant messages before they go upstream.

    Tags are for the human, not the model: if tagged history reaches the
    model it (a) imitates the tag (double-tagging) and (b) speculates about
    a 'routing layer' it cannot see. Returns a new payload with clean
    assistant content; other messages untouched.
    """
    # any bracketed or bare route-icon tag, plus exact configured tags
    tag_line = re.compile(r"^\s*(?:\[(?:🟢|⚡)[^\]]*\]|(?:🟢|⚡))\s*$")
    configured = {t.strip() for t in (TAG_LOCAL, TAG_ESC) if t and t.strip()}
    msgs = payload.get("messages")
    if not msgs:
        return payload
    changed = False
    clean = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant" \
                and isinstance(m.get("content"), str):
            c = m["content"]
            # strip any tag line (configured, renamed, or imitated)
            lines = [
                ln for ln in c.split("\n")
                if not tag_line.match(ln) and ln.strip() not in configured
            ]
            new_c = "\n".join(lines)
            if new_c != c:
                m = dict(m)
                m["content"] = new_c
                changed = True
        clean.append(m)
    if not changed:
        return payload
    out = dict(payload)
    out["messages"] = clean
    return out


def _needs_harness(payload):
    """True if the request involves Hermes tool machinery → pass through."""
    if payload.get("tools"):
        return True
    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("tool", "function"):
            return True
        if m.get("tool_calls") or m.get("function_call"):
            return True
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def _thread_decision_id(payload, decision_id):
    """Inject *decision_id* into the payload so downstream workers can echo it back.

    Writes a system message at the top of the messages list — this is the most
    reliable surface because it survives all transports (HTTP headers may be
    stripped by intermediaries, but a system-prompt stanza always reaches the model).
    """
    msgs = copy.deepcopy(payload.get("messages") or [])
    stanza = f"ROUTING DECISION ID: {decision_id} — include this id in any follow-up metadata."
    # Insert at position 0 so it's the first context the model sees.
    msgs.insert(0, {"role": "system", "content": stanza})
    payload["messages"] = msgs


def _judge(task_text):
    """Ask Laya via the proven hook client. Returns (p_hard, flag, source)."""
    j = laya_hook.apply(laya_hook.judge(task_text, sidecar_url=JUDGE_URL))
    return j.get("p_hard"), bool(j.get("difficulty_flag")), j.get("source", "laya")


def _decide_route(
    has_harness: bool,
    p_hard,
    flag: bool,
    source: str,
    fallback_route: str | None = None,
) -> tuple[str, str, str | None]:
    """Choose (decision, tag, rule) from judge output and fallback policy."""
    fb = (FALLBACK_ROUTE if fallback_route is None else fallback_route).strip().lower()
    if fb not in ("local", "escalate"):
        fb = "local"
    judge_failed = source in ("fallback", "judge-failed")
    if judge_failed:
        if fb == "escalate":
            return (
                "passthrough-harness" if has_harness else "escalate",
                TAG_ESC,
                "judge-failed-escalate",
            )
        return (
            "local-harness" if has_harness else "local",
            TAG_LOCAL,
            "judge-failed-local",
        )
    if flag:
        return ("passthrough-harness" if has_harness else "escalate", TAG_ESC, None)
    return ("local-harness" if has_harness else "local", TAG_LOCAL, None)


def _log(rec):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # logging must never break the chat


@app.get("/health")
def health():
    key_summary, warnings_list = _collect_startup_diagnostics()
    return {
        "ok": True,
        "proxy": "rovik",
        "local": LOCAL_URL,
        "judge": JUDGE_URL,
        "fallback_route": FALLBACK_ROUTE,
        "esc_failure_policy": ESC_FAILURE_POLICY,
        "esc_key": key_summary,
        "warnings": warnings_list,
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "rovik-proxy", "object": "model",
            "created": int(time.time()), "owned_by": "rovik"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    messages = payload.get("messages") or []
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Outcome-capture key — minted at routing time, threaded downstream.
    decision_id = uuid.uuid4().hex[:12]
    task_text_raw = _latest_user_text(messages)
    rec["decision_id"] = decision_id
    rec["task_text_hash"] = hashlib.sha256(task_text_raw.encode()).hexdigest()[:16]

    try:
        # 0. Route decision
        has_harness = _needs_harness(payload)
        task = task_text_raw
        rec["task"] = task[:200]
        try:
            p_hard, flag, source = _judge(task)
        except Exception:
            p_hard, flag, source = None, False, "judge-failed"
        rec["p_hard"], rec["source"] = p_hard, source
        decision, tag, rule = _decide_route(has_harness, p_hard, flag, source)
        rec["decision"] = decision
        rec["routed_to"] = "local" if decision in ("local", "local-harness") else "frontier"
        if rule:
            rec["rule"] = rule

        # 1. Route (local + local-harness both go to the 35B, tools forwarded)
        upstream = None
        esc_key_info = None
        if decision in ("local", "local-harness"):
            try:
                local_payload = _strip_tags_from_history(dict(payload))
                local_payload["model"] = LOCAL_MODEL
                local_payload.pop("chat_template_kwargs", None)  # llama-server args differ
                # Thread outcome-capture key into the request so it can be echoed back.
                _thread_decision_id(local_payload, decision_id)
                if not payload.get("stream"):
                    resp = _post(LOCAL_URL, local_payload,
                                 {"Content-Type": "application/json"}, 600)
                    resp["model"] = "rovik-local-35b"
                    content = resp["choices"][0]["message"].get("content") or ""
                    if tag:
                        resp["choices"][0]["message"]["content"] = f"{tag}\n{content}"
                    if isinstance(resp.get("usage"), dict):
                        rec["usage"] = resp["usage"]
                    rec["upstream"] = ["local", LOCAL_URL]
                    rec["answered_by"] = LOCAL_MODEL
                    _log(rec)
                    return JSONResponse(resp)
                # streaming: fall through to SSE relay with local upstream
                upstream = ("local", LOCAL_URL, local_payload,
                            {"Content-Type": "application/json"})
                rec["answered_by"] = LOCAL_MODEL
            except Exception as e:
                rec["local_error"] = str(e)[:120]
                rec["rule"] = "local-failed-escalate"
                upstream = None
                decision = "passthrough-harness" if has_harness else "escalate"
                rec["decision"] = decision
                rec["routed_to"] = "frontier"
                tag = TAG_ESC

        # 2. Escalation / pass-through: the session model, unchanged
        if upstream is None:
            esc_payload = _strip_tags_from_history(dict(payload))
            esc_payload["model"] = os.environ.get("ROVIK_ESC_MODEL", "z-ai/glm-5.3")
            esc_payload.pop("chat_template_kwargs", None)  # llama-server-specific
            # Thread outcome-capture key into the escalation request too.
            _thread_decision_id(esc_payload, decision_id)
            esc_key_info = _esc_key_info()
            upstream = ("esc", ESC_URL, esc_payload, _esc_headers())
            rec["answered_by"] = "session-model"

        kind, up_url, up_payload, up_headers = upstream
        rec["upstream"] = [kind, up_url]
        if payload.get("stream") and kind == "esc":
            up_payload.setdefault("stream_options", {})["include_usage"] = True

        # 3a. Streaming request → SSE relay (chunks relayed verbatim, tag prepended)
        if payload.get("stream"):

            def sse_relay():
                try:
                    req = urllib.request.Request(
                        up_url, data=json.dumps(up_payload).encode(),
                        headers=up_headers)
                    with urllib.request.urlopen(req, timeout=600) as r:
                        tag_sent = False
                        for line in r:
                            s = line.decode("utf-8", "replace")
                            if s.startswith("data: "):
                                body = s[6:].strip()
                                if body and body != "[DONE]":
                                    try:
                                        j = json.loads(body)
                                        if isinstance(j.get("usage"), dict):
                                            rec["usage"] = j["usage"]
                                        d = (j.get("choices") and j["choices"][0].get("delta")) or {}
                                        if not tag_sent and tag and d.get("content"):
                                            d["content"] = f"{tag}\n{d['content']}"
                                            tag_sent = True
                                            s = "data: " + json.dumps(j) + "\n\n"
                                    except ValueError:
                                        pass
                            yield s
                except urllib.error.HTTPError as exc:
                    try:
                        msg = _format_http_error(exc, key_info=esc_key_info if kind == "esc" else None)
                    finally:
                        try:
                            exc.close()
                        except Exception:
                            pass
                    rec["error"] = msg[:500]
                    err = json.dumps({"error": {"message": f"rovik relay failure: {msg}",
                                                "type": "proxy_error"}})
                    yield f"data: {err}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    rec["error"] = str(e)[:500]
                    err = json.dumps({"error": {"message": f"rovik relay failure: {e}",
                                                "type": "proxy_error"}})
                    yield f"data: {err}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    _log(rec)  # logged at stream end so token usage is included

            return StreamingResponse(sse_relay(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})

        # 3b. Non-streaming escalation
        try:
            resp = _post(up_url, up_payload, up_headers, 600)
        except UpstreamHTTPError as exc:
            msg = str(exc)
            if kind == "esc" and exc.status in (401, 403):
                msg = f"{msg}{_auth_key_hint(esc_key_info)}"
            raise UpstreamHTTPError(exc.status, msg, body=exc.body) from exc
        except urllib.error.HTTPError as exc:
            try:
                body = _read_http_error_body(exc)
                msg = _format_http_error(exc, key_info=esc_key_info if kind == "esc" else None, body=body)
                code = exc.code
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
            raise UpstreamHTTPError(code, msg, body=body) from exc

        if tag:
            content = resp["choices"][0]["message"].get("content") or ""
            resp["choices"][0]["message"]["content"] = f"{tag}\n{content}"
        if isinstance(resp.get("usage"), dict):
            rec["usage"] = resp["usage"]
        _log(rec)
        return JSONResponse(resp)

    except Exception as e:
        # Deliberate fail-loud on escalation/upstream failures (HTTP 502) so auth or provider errors surface cleanly.
        rec["error"] = str(e)[:500]
        _log(rec)
        return JSONResponse({"error": {"message": f"rovik proxy failure: {e}"}},
                            status_code=502)


def _read_log():
    recs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return recs


# Frontier price used for savings estimates: blended $ per 1M tokens you
# would have paid if every request went to the escalation model.
PRICE_PER_MTOK = float(os.environ.get("ROVIK_FRONTIER_PRICE_PER_MTOK", "3.0"))


@app.get("/stats")
def stats():
    recs = _read_log()
    routed = [r for r in recs if r.get("decision")]
    local = [r for r in routed if r.get("decision") in ("local", "local-harness")]
    esc = [r for r in routed if r.get("decision") in ("escalate", "passthrough-harness")]

    def _tok(rs, keys=("prompt_tokens", "completion_tokens")):
        n = 0
        for r in rs:
            u = r.get("usage") or {}
            n += sum(int(u.get(k) or 0) for k in keys)
        return n

    local_tok = _tok(local)
    esc_tok = _tok(esc)
    savings = local_tok / 1_000_000 * PRICE_PER_MTOK

    return {
        "decisions_total": len(routed),
        "local_count": len(local),
        "escalated_count": len(esc),
        "local_tokens": local_tok,
        "escalated_tokens": esc_tok,
        "savings_usd_est": round(savings, 4),
        "price_per_mtok_assumed": PRICE_PER_MTOK,
        "basis": "savings = tokens answered locally x frontier price; usage captured from upstream responses",
        "recent": routed[-10:],
    }


@app.get("/")
def dashboard():
    page = os.path.join(HERE, "dashboard.html")
    if os.path.exists(page):
        with open(page, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Rovik</h1><p>proxy is running; dashboard.html not found</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
