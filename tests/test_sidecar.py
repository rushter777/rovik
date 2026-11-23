"""Stdlib-only unit tests for serving/sidecar.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _install_sidecar_stubs():
    """Install minimal fastapi/uvicorn stubs only if not installed."""
    injected = []
    if importlib.util.find_spec("fastapi") is None:
        fastapi_mod = types.ModuleType("fastapi")
        responses_mod = types.ModuleType("fastapi.responses")

        class _FastAPI:
            def __init__(self, *args, **kwargs):
                self.lifespan = kwargs.get("lifespan")

            def get(self, path):
                return lambda fn: fn

            def post(self, path):
                return lambda fn: fn

        class _JSONResponse:
            def __init__(self, content=None, status_code=200, **kwargs):
                self.content = content
                self.status_code = status_code
                self.body = json.dumps(content).encode("utf-8")

        fastapi_mod.FastAPI = _FastAPI
        responses_mod.JSONResponse = _JSONResponse
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = responses_mod
        injected.extend(["fastapi", "fastapi.responses"])

    if importlib.util.find_spec("uvicorn") is None:
        uvicorn_mod = types.ModuleType("uvicorn")
        uvicorn_mod.run = lambda *args, **kwargs: None
        sys.modules["uvicorn"] = uvicorn_mod
        injected.append("uvicorn")

    return injected


_STUBBED_MODULES = []


def setUpModule():
    global _STUBBED_MODULES, sidecar, hook
    _STUBBED_MODULES = _install_sidecar_stubs()
    from serving import hook as _hook
    from serving import sidecar as _sidecar
    hook = _hook
    sidecar = _sidecar


def tearDownModule():
    if "serving.sidecar" in sys.modules and _STUBBED_MODULES:
        sys.modules.pop("serving.sidecar", None)
    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)


class _DummyAgent:
    def predict(self, state, questions):
        return {"answers": {"tools_file": {"type": "noul", "noul": 0.2}}}


class TestSidecar(unittest.TestCase):
    def test_question_set_matches_hook(self):
        self.assertEqual(sidecar.QUESTION_SET, hook.QUESTION_SET)

    def test_health_unloaded_and_loaded(self):
        with unittest.mock.patch.object(sidecar, "_agent", None), \
                unittest.mock.patch.object(sidecar, "_load_error", "missing ckpt"), \
                unittest.mock.patch.object(sidecar, "_startup_warnings", ["clamping temp"]):
            resp = sidecar.health()
            self.assertEqual(resp.status_code, 503)
            body = json.loads(resp.body.decode("utf-8"))
            self.assertFalse(body["ok"])
            self.assertEqual(body["error"], "missing ckpt")
            self.assertEqual(body["warnings"], ["clamping temp"])

        with unittest.mock.patch.object(sidecar, "_agent", _DummyAgent()), \
                unittest.mock.patch.object(sidecar, "_backend_name", "laya_mlx"), \
                unittest.mock.patch.object(sidecar, "_request_count", 7), \
                unittest.mock.patch.object(sidecar, "CLEAR_CACHE_EVERY", 10), \
                unittest.mock.patch.object(sidecar, "_startup_warnings", []):
            resp = sidecar.health()
            self.assertEqual(
                resp,
                {
                    "ok": True,
                    "model": sidecar.MODEL_PATH,
                    "backend": "laya_mlx",
                    "requests": 7,
                    "clear_cache_every": 10,
                    "warnings": [],
                },
            )

    def test_load_model_captures_runtime_warnings_and_exposes_on_health(self):
        warn_text = (
            "laya: this checkpoint ships temperatures outside [0.1, 10] which would distort "
            "confidence; clamping temperature[1]=12 -> 10. Treat confidence from the affected "
            "buckets as uncalibrated."
        )
        fake_laya = types.ModuleType("laya_mlx")

        def fake_load(_path):
            warnings.warn(warn_text, RuntimeWarning)
            return _DummyAgent()

        fake_laya.load = fake_load

        with tempfile.TemporaryDirectory() as tmpdir, \
                unittest.mock.patch.dict(sys.modules, {"laya_mlx": fake_laya}), \
                unittest.mock.patch.object(sidecar, "MODEL_PATH", tmpdir), \
                unittest.mock.patch.object(sidecar, "_agent", None), \
                unittest.mock.patch.object(sidecar, "_load_error", None), \
                unittest.mock.patch.object(sidecar, "_startup_warnings", []), \
                unittest.mock.patch.object(sidecar, "_backend_name", None), \
                unittest.mock.patch("sys.stderr", new_callable=lambda: open(os.devnull, "w")) as devnull:
            try:
                sidecar._load_model()
                h = sidecar.health()
            finally:
                devnull.close()
            self.assertTrue(h["ok"])
            self.assertEqual(h["backend"], "laya_mlx")
            self.assertIn(warn_text, h["warnings"])

    def test_load_model_cpu_fallback_and_missing_path_error(self):
        fake_cpu = types.ModuleType("laya")
        fake_cpu.load = lambda _path: _DummyAgent()

        with tempfile.TemporaryDirectory() as tmpdir, \
                unittest.mock.patch.object(sidecar, "_import_laya_backend", return_value=(fake_cpu, "laya")), \
                unittest.mock.patch.object(sidecar, "MODEL_PATH", tmpdir), \
                unittest.mock.patch.object(sidecar, "_agent", None), \
                unittest.mock.patch.object(sidecar, "_load_error", None), \
                unittest.mock.patch.object(sidecar, "_startup_warnings", []), \
                unittest.mock.patch.object(sidecar, "_backend_name", None):
            sidecar._load_model()
            h = sidecar.health()
            self.assertTrue(h["ok"])
            self.assertEqual(h["backend"], "laya")

        missing_dir = os.path.join(tempfile.gettempdir(), "nonexistent-rovik-ckpt-xyz")
        with unittest.mock.patch.object(sidecar, "_import_laya_backend", return_value=(fake_cpu, "laya")), \
                unittest.mock.patch.object(sidecar, "MODEL_PATH", missing_dir), \
                unittest.mock.patch.object(sidecar, "_agent", None), \
                unittest.mock.patch.object(sidecar, "_load_error", None), \
                unittest.mock.patch.object(sidecar, "_startup_warnings", []), \
                unittest.mock.patch.object(sidecar, "_backend_name", None), \
                unittest.mock.patch("sys.stderr", new_callable=lambda: open(os.devnull, "w")) as devnull:
            try:
                sidecar._load_model()
                resp = sidecar.health()
            finally:
                devnull.close()
            self.assertEqual(resp.status_code, 503)
            body = json.loads(resp.body.decode("utf-8"))
            self.assertIn("LAYA_MODEL_PATH does not exist", body["error"])

    def test_periodic_clear_mlx_cache_triggers_on_interval(self):
        with unittest.mock.patch.object(sidecar, "_agent", _DummyAgent()), \
                unittest.mock.patch.object(sidecar, "_request_count", 0), \
                unittest.mock.patch.object(sidecar, "CLEAR_CACHE_EVERY", 3), \
                unittest.mock.patch.object(sidecar, "_clear_mlx_cache") as m_clear:
            for i in range(6):
                out = sidecar.predict({"state": f"task {i}"})
                self.assertIn("answers", out)
            self.assertEqual(m_clear.call_count, 2)
            self.assertEqual(sidecar._request_count, 6)

    def test_clear_mlx_cache_disabled_when_zero(self):
        with unittest.mock.patch.object(sidecar, "_agent", _DummyAgent()), \
                unittest.mock.patch.object(sidecar, "_request_count", 0), \
                unittest.mock.patch.object(sidecar, "CLEAR_CACHE_EVERY", 0), \
                unittest.mock.patch.object(sidecar, "_clear_mlx_cache") as m_clear:
            for i in range(5):
                sidecar.predict({"state": f"task {i}"})
            self.assertEqual(m_clear.call_count, 0)


if __name__ == "__main__":
    unittest.main()
