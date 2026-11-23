"""Stdlib-only unit tests for rovik/config.py, rovik/init_wizard.py, and rovik/cli.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
import unittest.mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rovik import cli, config, harness, init_wizard, preflight, process  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_parse_write_roundtrip_and_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "sub", "rovik.env")
            vals = {
                "ROVIK_LOCAL_URL": "http://127.0.0.1:8082/v1/chat/completions",
                "ROVIK_TAG_LOCAL": '[🟢 LOCAL · "35B"]',
                "ROVIK_TAG_ESC": "[⚡ ESCALATED]",
                "EMPTY_VAR": "",
            }
            config.write_env_file(cfg_path, vals, header_comments=["Test config"])
            mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
            self.assertEqual(mode, 0o600)

            parsed = config.parse_env_file(cfg_path)
            self.assertEqual(parsed, vals)

    def test_parse_handles_comments_exports_and_single_quotes(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write("# header\n\nexport ROVIK_LOCAL_MODEL='qwen3.6'\nROVIK_PROXY_PORT=8790\n")
            path = tf.name
        try:
            parsed = config.parse_env_file(path)
            self.assertEqual(parsed["ROVIK_LOCAL_MODEL"], "qwen3.6")
            self.assertEqual(parsed["ROVIK_PROXY_PORT"], "8790")
        finally:
            os.unlink(path)

    def test_load_config_respects_env_precedence_and_rovik_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "custom.env")
            config.write_env_file(cfg_path, {
                "ROVIK_LOCAL_MODEL": "from-file",
                "ROVIK_PROXY_PORT": "8795",
            })
            base_env = {"ROVIK_CONFIG": cfg_path, "ROVIK_LOCAL_MODEL": "from-process-env"}
            with unittest.mock.patch.dict(os.environ, base_env, clear=True):
                resolved, applied = config.load_config()
                self.assertEqual(resolved, cfg_path)
                self.assertEqual(os.environ["ROVIK_LOCAL_MODEL"], "from-process-env")
                self.assertEqual(os.environ["ROVIK_PROXY_PORT"], "8795")
                self.assertNotIn("ROVIK_LOCAL_MODEL", applied)
                self.assertEqual(config.ENV_OVERRIDES["ROVIK_LOCAL_MODEL"], ("from-process-env", "from-file"))

    def test_diagnose_key_source_precedence_conflict_and_no_secret_leak(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            tf.write('OPENROUTER_API_KEY="sk-good-file-value"\n')
            env_path = tf.name
        try:
            d_env = config.diagnose_key_source(
                "OPENROUTER_API_KEY",
                env_path,
                env={"OPENROUTER_API_KEY": "sk-stale-shell-value"},
            )
            self.assertEqual(d_env["source"], "env")
            self.assertTrue(d_env["conflict"])
            self.assertIsNotNone(d_env["warning"])
            self.assertNotIn("sk-stale-shell-value", d_env["warning"])
            self.assertNotIn("sk-good-file-value", d_env["warning"])

            d_file = config.diagnose_key_source("OPENROUTER_API_KEY", env_path, env={})
            self.assertEqual(d_file["source"], "env-file")
            self.assertEqual(d_file["value"], "sk-good-file-value")
            self.assertFalse(d_file["conflict"])
            self.assertIsNone(d_file["warning"])

            d_missing = config.diagnose_key_source("OPENROUTER_API_KEY", "/no/such/file.env", env={})
            self.assertEqual(d_missing["source"], "missing")
            self.assertIsNotNone(d_missing["warning"])
        finally:
            os.unlink(env_path)


class TestInitWizard(unittest.TestCase):
    def test_local_teacher_preset_path_and_parameterized_harness_output(self):
        answers = iter([
            "1",                    # llama.cpp preset
            "",                     # default local URL (:8082)
            "qwen3.6-35b",          # local model
            "[LOCAL-CUSTOM]",       # custom local tag
            "1",                    # OpenRouter preset
            "",                     # default frontier URL
            "",                     # default frontier model
            "",                     # default keyname
            "",                     # default env file
            "",                     # default price
            "[ESC-CUSTOM]",         # custom esc tag
            "1",                    # teacher = same as local
            "data/my_tasks.txt",    # corpus path
            "",                     # checkpoint
            "0.7",                  # threshold
            "local",                # fallback route
            "",                     # log dir
            "",                     # proxy host
            "9099",                 # custom proxy port
            "",                     # sidecar port
            "2",                    # harness = OpenCode
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "rovik.env")
            logs: list[str] = []
            vals = init_wizard.run_wizard(
                output_path=out_path,
                input_fn=lambda _p: next(answers),
                print_fn=logs.append,
            )
            self.assertEqual(vals["ROVIK_LOCAL_URL"], "http://127.0.0.1:8082/v1/chat/completions")
            self.assertEqual(vals["ROVIK_LOCAL_MODEL"], "qwen3.6-35b")
            self.assertEqual(vals["ROVIK_TEACHER_URL"], vals["ROVIK_LOCAL_URL"])
            self.assertEqual(vals["ROVIK_TEACHER_KEYNAME"], "")
            self.assertEqual(vals["ROVIK_CORPUS_PATH"], "data/my_tasks.txt")
            self.assertEqual(vals["LAYA_THRESH_HARD"], "0.7")
            self.assertEqual(vals["ROVIK_PROXY_PORT"], "9099")
            self.assertTrue(os.path.isfile(out_path))

            joined = "\n".join(logs)
            self.assertIn("Preflight check:", joined)
            self.assertIn("rovik start", joined)
            self.assertIn("rovik status", joined)
            self.assertIn("http://127.0.0.1:9099/v1", joined)
            self.assertNotIn("http://127.0.0.1:8788/v1", joined)
            self.assertIn("[LOCAL-CUSTOM]", joined)
            self.assertIn("[ESC-CUSTOM]", joined)

    def test_frontier_teacher_preset_and_rerun_preserves_defaults(self):
        answers = iter([
            "2",    # Ollama preset
            "", "", "",
            "2",    # OpenAI preset
            "", "", "", "", "", "",
            "2",    # teacher = same as frontier
            "", "", "0.5", "", "", "", "", "",
            "",     # default harness (Hermes)
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "rovik.env")
            vals = init_wizard.run_wizard(
                output_path=out_path,
                input_fn=lambda _p: next(answers),
                print_fn=lambda _s: None,
            )
            self.assertEqual(vals["ROVIK_LOCAL_URL"], "http://127.0.0.1:11434/v1/chat/completions")
            self.assertEqual(vals["ROVIK_ESC_URL"], "https://api.openai.com/v1/chat/completions")
            self.assertEqual(vals["ROVIK_TEACHER_URL"], "https://api.openai.com/v1/chat/completions")
            self.assertEqual(vals["ROVIK_TEACHER_KEYNAME"], "OPENAI_API_KEY")
            self.assertEqual(vals["LAYA_THRESH_HARD"], "0.5")

            # Pressing Enter through everything on re-run keeps every existing choice.
            rerun_vals = init_wizard.run_wizard(
                output_path=out_path,
                input_fn=lambda _p: "",
                print_fn=lambda _s: None,
            )
            self.assertEqual(rerun_vals, vals)

    def test_non_interactive_writes_valid_file_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "rovik.env")
            vals = init_wizard.run_wizard(
                output_path=out_path,
                non_interactive=True,
                print_fn=lambda _s: None,
            )
            self.assertEqual(vals["ROVIK_ESC_KEYNAME"], "OPENROUTER_API_KEY")
            with open(out_path, "r", encoding="utf-8") as f:
                raw = f.read()
            self.assertNotIn("sk-", raw)


class TestHarnessAndPreflight(unittest.TestCase):
    def test_render_harness_snippets_parameterized(self):
        for name in harness.HARNESS_ORDER + ("all",):
            txt = harness.render_harness_snippet(
                name,
                proxy_host="127.0.0.1",
                proxy_port=9123,
                local_model="my-qwen-35b",
                tag_local="[LOC]",
                tag_esc="[ESC]",
            )
            self.assertIn("http://127.0.0.1:9123/v1", txt)
            self.assertIn("Expected reply tags: local = [LOC] | escalated = [ESC]", txt)

        hermes_txt = harness.render_harness_snippet("hermes", proxy_port=9123, local_model="my-qwen-35b")
        self.assertIn("default: my-qwen-35b", hermes_txt)

        with self.assertRaises(ValueError):
            harness.render_harness_snippet("unknown-tool")

    def test_preflight_checks_backend_and_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch.object(preflight, "check_laya_backend", return_value=None), \
                    unittest.mock.patch.object(preflight, "check_serving_runtime", return_value=["fastapi"]):
                bad = preflight.run_preflight(model_path=os.path.join(tmpdir, "missing"), check_serving=True)
            self.assertFalse(bad.ok)
            self.assertEqual(len(bad.errors), 3)
            msg = preflight.format_preflight(bad)
            self.assertIn("laya_mlx", msg)
            self.assertIn("LAYA_MODEL_PATH", msg)
            self.assertIn("fastapi", msg)

            with unittest.mock.patch.object(preflight, "check_laya_backend", return_value="laya"), \
                    unittest.mock.patch.object(preflight, "check_serving_runtime", return_value=[]):
                good = preflight.run_preflight(model_path=tmpdir, check_serving=True)
            self.assertTrue(good.ok)
            self.assertEqual(good.laya_backend, "laya")
            self.assertEqual(good.errors, [])


class TestProcessLifecycle(unittest.TestCase):
    def test_start_refuses_when_preflight_fails(self):
        logs: list[str] = []
        fail_res = preflight.PreflightResult(
            ok=False,
            laya_backend=None,
            model_path="checkpoints/cold-test",
            model_exists=False,
            missing_runtime=[],
            errors=["Missing Laya inference package"],
        )
        with unittest.mock.patch.object(process, "run_preflight", return_value=fail_res), \
                unittest.mock.patch.object(process.subprocess, "Popen") as m_popen:
            rc = process.start_services(print_fn=logs.append)
        self.assertEqual(rc, 1)
        m_popen.assert_not_called()
        self.assertIn("refusing to start", "\n".join(logs))

    def test_start_stop_and_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "ROVIK_LOG_DIR": tmpdir,
                "LAYA_SIDECAR_HOST": "127.0.0.1",
                "LAYA_SIDECAR_PORT": "18777",
                "ROVIK_PROXY_HOST": "127.0.0.1",
                "ROVIK_PROXY_PORT": "18788",
            }
            fake_procs = []

            def fake_popen(cmd, **_kwargs):
                m = unittest.mock.MagicMock()
                m.pid = 41000 + len(fake_procs)
                m.poll.return_value = None
                fake_procs.append((cmd, m))
                return m

            probe_calls = {"sidecar": 0, "proxy": 0}

            def fake_probe(url, timeout_s=2.0):
                if ":18777/health" in url:
                    probe_calls["sidecar"] += 1
                    if probe_calls["sidecar"] == 1:
                        return None, None, "connection refused"
                    return 200, {
                        "ok": True,
                        "model": "checkpoints/cold-test",
                        "backend": "laya_mlx",
                        "requests": 1,
                        "warnings": ["clamping temperature[1]: treat confidence as uncalibrated"],
                    }, None
                if ":18788/health" in url:
                    probe_calls["proxy"] += 1
                    if probe_calls["proxy"] == 1:
                        return None, None, "connection refused"
                    return 200, {
                        "ok": True,
                        "proxy": "rovik",
                        "local": "http://127.0.0.1:8082/v1/chat/completions",
                        "judge": "http://127.0.0.1:18777/predict",
                        "fallback_route": "local",
                        "esc_failure_policy": "fail-loud",
                        "esc_key": {"name": "OPENROUTER_API_KEY", "source": "env", "env_file": "", "conflict": True},
                        "warnings": ["OPENROUTER_API_KEY conflict warning"],
                    }, None
                if ":18788/stats" in url:
                    return 200, {"decisions_total": 3, "local_count": 2, "escalated_count": 1, "savings_usd_est": 0.01}, None
                return None, None, "unexpected"

            logs: list[str] = []
            ok_pf = preflight.PreflightResult(
                ok=True, laya_backend="laya_mlx", model_path="checkpoints/cold-test", model_exists=True
            )
            with unittest.mock.patch.dict(os.environ, env, clear=True), \
                    unittest.mock.patch.object(process, "run_preflight", return_value=ok_pf), \
                    unittest.mock.patch.object(process, "_find_pids_by_port", return_value=[]), \
                    unittest.mock.patch.object(process, "_probe_json", side_effect=fake_probe), \
                    unittest.mock.patch.object(process.subprocess, "Popen", side_effect=fake_popen):
                rc = process.start_services(base_env=env, wait=True, timeout_s=5.0, print_fn=logs.append)
                self.assertEqual(rc, 0)
                self.assertEqual(len(fake_procs), 2)
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "sidecar.pid")))
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "proxy.pid")))
                out = "\n".join(logs)
                self.assertIn("Treat confidence as uncalibrated".lower(), out.lower())
                self.assertIn("OPENROUTER_API_KEY conflict warning", out)

                status_logs: list[str] = []
                with unittest.mock.patch.object(process, "_pid_alive", side_effect=lambda p: p in (41000, 41001)):
                    st_rc = process.status_services(print_fn=status_logs.append)
                self.assertEqual(st_rc, 0)
                st_out = "\n".join(status_logs)
                self.assertIn("Sidecar: healthy", st_out)
                self.assertIn("Proxy:   healthy", st_out)
                self.assertIn("clamping temperature[1]: treat confidence as uncalibrated", st_out)
                self.assertIn("OPENROUTER_API_KEY conflict warning", st_out)

                # Stop: pidfile stops 41001 (proxy), while 41000 pidfile removed & stopped via port fallback
                os.unlink(os.path.join(tmpdir, "sidecar.pid"))
                stop_logs: list[str] = []
                terminated: list[int] = []

                def fake_port_pids(port):
                    return [41000] if port == 18777 else []

                with unittest.mock.patch.object(process, "_pid_alive", return_value=True), \
                        unittest.mock.patch.object(process, "_find_pids_by_port", side_effect=fake_port_pids), \
                        unittest.mock.patch.object(process, "_is_rovik_process", return_value=True), \
                        unittest.mock.patch.object(process, "_terminate_pid", side_effect=lambda p, timeout_s=5.0: terminated.append(p) or True):
                    stop_rc = process.stop_services(print_fn=stop_logs.append)
                self.assertEqual(stop_rc, 0)
                self.assertEqual(terminated, [41001, 41000])
                self.assertFalse(os.path.exists(os.path.join(tmpdir, "proxy.pid")))
                self.assertIn("stopped by port 18777", "\n".join(stop_logs))

    def test_status_returns_nonzero_when_sidecar_503(self):
        def fake_probe(url, timeout_s=2.0):
            if ":8777/health" in url:
                return 503, {"ok": False, "error": "missing ckpt", "warnings": ["bad temp"]}, "HTTP 503"
            return None, None, "connection refused"

        logs: list[str] = []
        with unittest.mock.patch.dict(
            os.environ,
            {"LAYA_SIDECAR_PORT": "8777", "ROVIK_PROXY_PORT": "8788"},
            clear=True,
        ), unittest.mock.patch.object(process, "_find_pids_by_port", return_value=[]), \
                unittest.mock.patch.object(process, "_probe_json", side_effect=fake_probe):
            rc = process.status_services(print_fn=logs.append)
        self.assertEqual(rc, 1)
        out = "\n".join(logs)
        self.assertIn("unloaded (503)", out)
        self.assertIn("bad temp", out)


class TestCLI(unittest.TestCase):
    def test_config_command_loads_file_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "rovik.env")
            fallback_env = os.path.join(tmpdir, "hermes.env")
            config.write_env_file(fallback_env, {"OPENROUTER_API_KEY": "sk-file-key"})
            config.write_env_file(cfg_path, {
                "ROVIK_LOCAL_MODEL": "qwen-test",
                "ROVIK_ESC_KEYNAME": "OPENROUTER_API_KEY",
                "ROVIK_ENV_FILE": fallback_env,
            })
            buf = io.StringIO()
            with unittest.mock.patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "super-secret-value", "ROVIK_LOCAL_MODEL": "qwen-override"},
                clear=True,
            ), contextlib.redirect_stdout(buf):
                rc = cli.main(["--config", cfg_path, "config"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("qwen-override", out)
            self.assertIn("overrides", out)
            self.assertIn("OPENROUTER_API_KEY (env: set)", out)
            self.assertIn("differs from", out)
            self.assertNotIn("super-secret-value", out)
            self.assertNotIn("sk-file-key", out)

    def test_harness_command_uses_loaded_config_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "rovik.env")
            config.write_env_file(cfg_path, {
                "ROVIK_PROXY_PORT": "9222",
                "ROVIK_LOCAL_MODEL": "local-custom",
                "ROVIK_TAG_LOCAL": "[MY-LOCAL]",
                "ROVIK_TAG_ESC": "[MY-ESC]",
            })
            buf = io.StringIO()
            with unittest.mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(buf):
                rc = cli.main(["--config", cfg_path, "harness", "opencode"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("http://127.0.0.1:9222/v1", out)
            self.assertIn("[MY-LOCAL]", out)
            self.assertIn("[MY-ESC]", out)

    def test_status_json_dispatch(self):
        buf = io.StringIO()
        fake_status = {"ok": False, "sidecar": {"ok": False}, "proxy": {"ok": False}, "warnings": ["w1"]}
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
                unittest.mock.patch("rovik.process.collect_status", return_value=fake_status), \
                contextlib.redirect_stdout(buf):
            rc = cli.main(["status", "--json"])
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(buf.getvalue()), fake_status)

    def test_judge_dispatch_and_unknown_command(self):
        buf = io.StringIO()
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
                unittest.mock.patch("serving.hook.judge", return_value=None) as m_judge, \
                contextlib.redirect_stdout(buf):
            rc = cli.main(["judge", "hello", "world"])
        self.assertEqual(rc, 0)
        m_judge.assert_called_once_with("hello world")
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["source"], "fallback")

        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            bad_rc = cli.main(["no-such-cmd"])
        self.assertNotEqual(bad_rc, 0)


if __name__ == "__main__":
    unittest.main()
