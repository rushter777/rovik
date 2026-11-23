"""Stdlib-only unit tests for serving/proxy.py."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
import urllib.error

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
            self.lifespan = kwargs.get("lifespan")

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
            if isinstance(content, str):
                self.body = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                self.body = bytes(content)
            else:
                self.body = b""

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


async def _consume_streaming(resp):
    if hasattr(resp, "body_iterator"):
        chunks = []
        async for part in resp.body_iterator:
            chunks.append(part.decode("utf-8", "replace") if isinstance(part, bytes) else str(part))
        return "".join(chunks)
    return "".join(resp.content)


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

    def test_startup_diagnostics_logs_source_conflict_and_env_overrides(self):
        import io as _io

        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = os.path.join(tmpdir, ".hermes.env")
            cfg_file = os.path.join(tmpdir, "rovik.env")
            with open(env_file, "w", encoding="utf-8") as f:
                f.write("OPENROUTER_API_KEY=sk-good-file-secret\n")
            with open(cfg_file, "w", encoding="utf-8") as f:
                f.write("ROVIK_LOCAL_MODEL=from-cfg\n")

            buf = _io.StringIO()
            env_vals = {
                "OPENROUTER_API_KEY": "sk-stale-env-secret",
                "ROVIK_LOCAL_MODEL": "from-shell-env",
            }
            with unittest.mock.patch.dict(os.environ, env_vals, clear=True), \
                    unittest.mock.patch.object(proxy, "ENV_FILE", env_file), \
                    unittest.mock.patch.object(proxy, "ESC_KEYNAME", "OPENROUTER_API_KEY"), \
                    unittest.mock.patch.object(proxy.rovik_config, "LOADED_CONFIG_PATH", cfg_file), \
                    unittest.mock.patch.object(proxy.rovik_config, "ENV_OVERRIDES", {}):
                warns = proxy.log_startup_diagnostics(stream=buf)

            text = buf.getvalue()
            self.assertIn("source=env", text)
            self.assertIn("OPENROUTER_API_KEY in shell environment overrides", text)
            self.assertIn("ROVIK_LOCAL_MODEL in process environment", text)
            self.assertGreaterEqual(len(warns), 2)
            self.assertNotIn("sk-stale-env-secret", text)
            self.assertNotIn("sk-good-file-secret", text)


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

    def test_chat_famous_but_simple_prompt_below_cutoff_stays_local(self):
        posted_urls = []
        logged_recs = []

        def fake_judge(_task, sidecar_url=None):
            return {
                "tools_file": {"type": "noul", "noul": 0.04},
                "tools_terminal": {"type": "noul", "noul": 0.02},
                "tools_web": {"type": "noul", "noul": 0.12},
                "needs_verification": {"type": "noul", "noul": 0.15},
                "difficulty": {"probabilities": {"0": 0.72, "1": 0.18, "2": 0.06, "3": 0.03, "4": 0.01}},
            }

        def fake_post(url, payload, headers, timeout_s=300):
            posted_urls.append(url)
            return {
                "id": "chatcmpl-riemann",
                "choices": [{"message": {"role": "assistant", "content": "The Riemann hypothesis concerns zeros of zeta(s)."}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 15},
            }

        req = _FakeRequest({
            "messages": [{"role": "user", "content": "Explain the Riemann hypothesis in two sentences."}],
            "stream": False,
        })
        with unittest.mock.patch.object(proxy.laya_hook, "judge", side_effect=fake_judge), \
                unittest.mock.patch.object(proxy, "_post", side_effect=fake_post), \
                unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append):
            resp = asyncio.run(proxy.chat(req))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(posted_urls, [proxy.LOCAL_URL])
        body = json.loads(resp.body.decode("utf-8"))
        self.assertNotIn("error", body)
        self.assertTrue(body["choices"][0]["message"]["content"].startswith(proxy.TAG_LOCAL))
        self.assertEqual(logged_recs[0]["decision"], "local")
        self.assertEqual(logged_recs[0]["routed_to"], "local")
        self.assertAlmostEqual(logged_recs[0]["p_hard"], 0.28, places=4)

    def test_chat_judge_down_falls_back_to_local_with_zero_client_error(self):
        for mode in ("returns_none", "raises"):
            posted_urls = []
            logged_recs = []

            def fake_post(url, payload, headers, timeout_s=300):
                posted_urls.append(url)
                return {
                    "id": "chatcmpl-fallback",
                    "choices": [{"message": {"role": "assistant", "content": "Local answer"}}],
                }

            req = _FakeRequest({"messages": [{"role": "user", "content": "Summarize this paragraph"}], "stream": False})
            if mode == "returns_none":
                patch_judge = unittest.mock.patch.object(proxy.laya_hook, "judge", return_value=None)
            else:
                patch_judge = unittest.mock.patch.object(proxy, "_judge", side_effect=RuntimeError("sidecar down"))

            with patch_judge, \
                    unittest.mock.patch.object(proxy, "FALLBACK_ROUTE", "local"), \
                    unittest.mock.patch.object(proxy, "_post", side_effect=fake_post), \
                    unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append):
                resp = asyncio.run(proxy.chat(req))

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(posted_urls, [proxy.LOCAL_URL])
            body = json.loads(resp.body.decode("utf-8"))
            self.assertNotIn("error", body)
            self.assertTrue(body["choices"][0]["message"]["content"].startswith(proxy.TAG_LOCAL))
            self.assertEqual(logged_recs[0]["rule"], "judge-failed-local")

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

    def test_chat_frontier_401_fails_loudly_with_upstream_body_and_key_hint(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write('OPENROUTER_API_KEY="sk-good-file-secret"\n')
            env_path = tf.name
        logged_recs = []

        def fake_urlopen(req, timeout=300):
            raise urllib.error.HTTPError(
                proxy.ESC_URL,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"error":{"message":"User not found.","code":401}}'),
            )

        try:
            req = _FakeRequest({"messages": [{"role": "user", "content": "Design distributed consensus"}], "stream": False})
            with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-stale-env-secret"}, clear=True), \
                    unittest.mock.patch.object(proxy, "ENV_FILE", env_path), \
                    unittest.mock.patch.object(proxy, "ESC_KEYNAME", "OPENROUTER_API_KEY"), \
                    unittest.mock.patch.object(proxy, "_judge", return_value=(0.95, True, "laya")), \
                    unittest.mock.patch.object(proxy.urllib.request, "urlopen", side_effect=fake_urlopen), \
                    unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append):
                resp = asyncio.run(proxy.chat(req))
        finally:
            os.unlink(env_path)

        self.assertEqual(resp.status_code, 502)
        body = json.loads(resp.body.decode("utf-8"))
        msg = body["error"]["message"]
        self.assertIn("HTTP Error 401: Unauthorized", msg)
        self.assertIn("User not found.", msg)
        self.assertIn("loaded from env; differs from", msg)
        self.assertNotIn("sk-stale-env-secret", msg)
        self.assertNotIn("sk-good-file-secret", msg)
        self.assertEqual(logged_recs[0]["upstream"], ["esc", proxy.ESC_URL])
        self.assertIn("User not found.", logged_recs[0]["error"])

    def test_chat_frontier_timeout_fails_loudly(self):
        posted_urls = []
        logged_recs = []

        def fake_post(url, payload, headers, timeout_s=300):
            posted_urls.append(url)
            raise TimeoutError("frontier timed out")

        req = _FakeRequest({"messages": [{"role": "user", "content": "Hard multi-step refactor"}], "stream": False})
        with unittest.mock.patch.object(proxy, "_judge", return_value=(0.95, True, "laya")), \
                unittest.mock.patch.object(proxy, "_post", side_effect=fake_post), \
                unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append), \
                unittest.mock.patch.object(proxy, "_esc_headers", return_value={"Content-Type": "application/json"}):
            resp = asyncio.run(proxy.chat(req))

        self.assertEqual(resp.status_code, 502)
        self.assertEqual(posted_urls, [proxy.ESC_URL])
        body = json.loads(resp.body.decode("utf-8"))
        self.assertIn("frontier timed out", body["error"]["message"])
        self.assertIn("frontier timed out", logged_recs[0]["error"])

    def test_chat_streaming_frontier_401_emits_upstream_body_in_sse_error(self):
        logged_recs = []

        def fake_urlopen(req, timeout=600):
            raise urllib.error.HTTPError(
                proxy.ESC_URL,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"error":{"message":"User not found."}}'),
            )

        req = _FakeRequest({"messages": [{"role": "user", "content": "Hard streaming task"}], "stream": True})
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
                unittest.mock.patch.object(proxy, "ENV_FILE", "/nonexistent/.env"), \
                unittest.mock.patch.object(proxy, "_judge", return_value=(0.95, True, "laya")), \
                unittest.mock.patch.object(proxy.urllib.request, "urlopen", side_effect=fake_urlopen), \
                unittest.mock.patch.object(proxy, "_log", side_effect=logged_recs.append):
            resp = asyncio.run(proxy.chat(req))
            stream_out = asyncio.run(_consume_streaming(resp))

        self.assertIn("User not found.", stream_out)
        self.assertIn("source: missing", stream_out)
        self.assertIn("data: [DONE]", stream_out)
        self.assertEqual(len(logged_recs), 1)
        self.assertIn("User not found.", logged_recs[0]["error"])


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


class TestEndpoints(unittest.TestCase):
    def test_health_and_dashboard_html(self):
        health = proxy.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["proxy"], "rovik")
        self.assertIn("fallback_route", health)
        self.assertEqual(health["esc_failure_policy"], "fail-loud")
        self.assertIn("esc_key", health)
        self.assertNotIn("value", health["esc_key"])
        self.assertIsInstance(health["warnings"], list)

        resp = proxy.dashboard()
        self.assertEqual(resp.status_code, 200)
        html = resp.body.decode("utf-8") if isinstance(getattr(resp, "body", None), (bytes, bytearray)) else resp.content
        self.assertIn("ROVIK", html)
        self.assertIn('<meta name="color-scheme"', html)
        self.assertIn('role="region"', html)
        self.assertIn("@container", html)
        self.assertIn("forced-colors", html)


if __name__ == "__main__":
    unittest.main()
