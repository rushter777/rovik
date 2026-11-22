"""Stdlib-only unit tests for serving/proxy.py."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _install_fastapi_stub():
    """Install minimal fastapi stubs only if fastapi is not installed."""
    injected = []
    if importlib.util.find_spec("fastapi") is not None:
        return injected

    fastapi_mod = types.ModuleType("fastapi")
    responses_mod = types.ModuleType("fastapi.responses")

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, path):
            return lambda fn: fn

        def post(self, path):
            return lambda fn: fn

    class _Request:
        pass

    class _Response:
        def __init__(self, content=None, status_code=200, **kwargs):
            self.content = content
            self.status_code = status_code

    class _JSONResponse(_Response):
        def __init__(self, content=None, status_code=200, **kwargs):
            super().__init__(content=content, status_code=status_code, **kwargs)
            self.body = json.dumps(content).encode("utf-8")

    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Request = _Request
    responses_mod.HTMLResponse = _Response
    responses_mod.JSONResponse = _JSONResponse
    responses_mod.StreamingResponse = _Response

    sys.modules["fastapi"] = fastapi_mod
    sys.modules["fastapi.responses"] = responses_mod
    injected.extend(["fastapi", "fastapi.responses"])
    return injected


_STUBBED_MODULES = []


def setUpModule():
    global _STUBBED_MODULES, proxy
    _STUBBED_MODULES = _install_fastapi_stub()
    from serving import proxy as _proxy
    proxy = _proxy


def tearDownModule():
    if "serving.proxy" in sys.modules and _STUBBED_MODULES:
        sys.modules.pop("serving.proxy", None)
    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestEscHeaders(unittest.TestCase):
    def test_env_precedence_over_env_file(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write('OPENROUTER_API_KEY="from-file"\n')
            env_path = tf.name
        try:
            with unittest.mock.patch.dict(
                os.environ, {"OPENROUTER_API_KEY": "from-env"}, clear=False
            ), unittest.mock.patch.object(proxy, "ENV_FILE", env_path), \
                    unittest.mock.patch.object(proxy, "ESC_KEYNAME", "OPENROUTER_API_KEY"):
                headers = proxy._esc_headers()
            self.assertEqual(headers.get("Authorization"), "Bearer from-env")
        finally:
            os.unlink(env_path)

    def test_env_file_fallback_parses_export_and_quotes(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write("# comment line\n\nexport OPENROUTER_API_KEY='quoted-secret'\n")
            env_path = tf.name
        try:
            env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
            with unittest.mock.patch.dict(os.environ, env_without_key, clear=True), \
                    unittest.mock.patch.object(proxy, "ENV_FILE", env_path), \
                    unittest.mock.patch.object(proxy, "ESC_KEYNAME", "OPENROUTER_API_KEY"):
                headers = proxy._esc_headers()
            self.assertEqual(headers.get("Authorization"), "Bearer quoted-secret")
        finally:
            os.unlink(env_path)

    def test_missing_key_omits_authorization(self):
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with unittest.mock.patch.dict(os.environ, env_without_key, clear=True), \
                unittest.mock.patch.object(proxy, "ENV_FILE", "/nonexistent/.env.rovik"), \
                unittest.mock.patch.object(proxy, "ESC_KEYNAME", "OPENROUTER_API_KEY"):
            headers = proxy._esc_headers()
        self.assertNotIn("Authorization", headers)


class TestStripTags(unittest.TestCase):
    def test_strips_bracketed_bare_and_custom_tags_only_from_assistant(self):
        payload = {
            "messages": [
                {"role": "system", "content": "[🟢 LOCAL · 35B-A3B]\nkeep system"},
                {"role": "user", "content": "🟢\nCUSTOM_LOCAL\nkeep user"},
                {
                    "role": "assistant",
                    "content": "[🟢 LOCAL · 35B-A3B]\n🟢\n⚡\n[⚡ ESCALATED]\nCUSTOM_LOCAL\nCUSTOM_ESC\nActual reply",
                },
            ]
        }
        with unittest.mock.patch.object(proxy, "TAG_LOCAL", "CUSTOM_LOCAL"), \
                unittest.mock.patch.object(proxy, "TAG_ESC", "CUSTOM_ESC"):
            cleaned = proxy._strip_tags_from_history(payload)
        self.assertEqual(cleaned["messages"][0]["content"], "[🟢 LOCAL · 35B-A3B]\nkeep system")
        self.assertEqual(cleaned["messages"][1]["content"], "🟢\nCUSTOM_LOCAL\nkeep user")
        self.assertEqual(cleaned["messages"][2]["content"], "Actual reply")


class TestRouting(unittest.TestCase):
    def test_decide_route_normal_and_harness(self):
        self.assertEqual(proxy._decide_route(False, 0.1, False, "laya"), ("local", proxy.TAG_LOCAL, None))
        self.assertEqual(proxy._decide_route(False, 0.9, True, "laya"), ("escalate", proxy.TAG_ESC, None))
        self.assertEqual(proxy._decide_route(True, 0.1, False, "laya"), ("local-harness", proxy.TAG_LOCAL, None))
        self.assertEqual(proxy._decide_route(True, 0.9, True, "laya"), ("passthrough-harness", proxy.TAG_ESC, None))

    def test_decide_route_judge_failure_respects_fallback_route(self):
        for src in ("fallback", "judge-failed"):
            self.assertEqual(
                proxy._decide_route(False, None, False, src, fallback_route="local"),
                ("local", proxy.TAG_LOCAL, "judge-failed-local"),
            )
            self.assertEqual(
                proxy._decide_route(True, None, False, src, fallback_route="local"),
                ("local-harness", proxy.TAG_LOCAL, "judge-failed-local"),
            )
            self.assertEqual(
                proxy._decide_route(False, None, False, src, fallback_route="escalate"),
                ("escalate", proxy.TAG_ESC, "judge-failed-escalate"),
            )
            self.assertEqual(
                proxy._decide_route(True, None, False, src, fallback_route="escalate"),
                ("passthrough-harness", proxy.TAG_ESC, "judge-failed-escalate"),
            )

    def test_judge_forwards_judge_url(self):
        with unittest.mock.patch.object(proxy, "JUDGE_URL", "http://127.0.0.1:9999/predict"), \
                unittest.mock.patch.object(proxy.laya_hook, "judge", return_value=None) as m_judge:
            p_hard, flag, source = proxy._judge("check disk space")
        m_judge.assert_called_once_with("check disk space", sidecar_url="http://127.0.0.1:9999/predict")
        self.assertIsNone(p_hard)
        self.assertFalse(flag)
        self.assertEqual(source, "fallback")

    def test_chat_local_error_falls_back_to_escalation(self):
        posted_urls = []
        logged_recs = []

        def fake_post(url, payload, headers, timeout_s=300):
            posted_urls.append(url)
            if url == proxy.LOCAL_URL:
                raise ConnectionError("local server down")
            return {
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "Recovered via frontier"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        req = _FakeRequest({"messages": [{"role": "user", "content": "Say hello"}], "stream": False})
        with unittest.mock.patch.object(proxy, "_judge", return_value=(0.1, False, "laya")), \
                unittest.mock.patch.object(proxy, "_post", side_effect=fake_post), \
                unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append), \
                unittest.mock.patch.object(proxy, "_esc_headers", return_value={"Content-Type": "application/json"}):
            resp = asyncio.run(proxy.chat(req))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(posted_urls, [proxy.LOCAL_URL, proxy.ESC_URL])
        body = json.loads(resp.body.decode("utf-8"))
        self.assertTrue(body["choices"][0]["message"]["content"].startswith(proxy.TAG_ESC))
        self.assertEqual(len(logged_recs), 1)
        rec = logged_recs[0]
        self.assertEqual(rec["decision"], "escalate")
        self.assertEqual(rec["routed_to"], "frontier")
        self.assertEqual(rec["rule"], "local-failed-escalate")
        self.assertEqual(rec["upstream"], ["esc", proxy.ESC_URL])


class TestLogPaths(unittest.TestCase):
    def test_default_and_override_log_file(self):
        expected_default = os.path.join(_PROJECT_ROOT, "data", "proxy_log.jsonl")
        env_no_dir = {k: v for k, v in os.environ.items() if k != "ROVIK_LOG_DIR"}
        try:
            with unittest.mock.patch.dict(os.environ, env_no_dir, clear=True):
                reloaded = importlib.reload(proxy)
                self.assertEqual(reloaded.LOG_FILE, expected_default)
            with tempfile.TemporaryDirectory() as tmpdir:
                with unittest.mock.patch.dict(os.environ, {"ROVIK_LOG_DIR": tmpdir}, clear=False):
                    reloaded = importlib.reload(proxy)
                    self.assertEqual(reloaded.LOG_FILE, os.path.join(tmpdir, "proxy_log.jsonl"))
        finally:
            importlib.reload(proxy)


if __name__ == "__main__":
    unittest.main()
