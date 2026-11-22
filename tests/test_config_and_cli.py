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

from rovik import cli, config, init_wizard  # noqa: E402


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


class TestInitWizard(unittest.TestCase):
    def test_local_teacher_preset_path(self):
        answers = iter([
            "1",                    # llama.cpp preset
            "",                     # default local URL (:8082)
            "qwen3.6-35b",          # local model
            "",                     # default local tag
            "1",                    # OpenRouter preset
            "",                     # default frontier URL
            "",                     # default frontier model
            "",                     # default keyname
            "",                     # default env file
            "",                     # default price
            "",                     # default esc tag
            "1",                    # teacher = same as local
            "data/my_tasks.txt",    # corpus path
            "",                     # checkpoint
            "0.7",                  # threshold
            "local",                # fallback route
            "",                     # log dir
            "",                     # proxy host
            "",                     # proxy port
            "",                     # sidecar port
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
            self.assertTrue(os.path.isfile(out_path))

    def test_frontier_teacher_preset_and_rerun_preserves_defaults(self):
        answers = iter([
            "2",    # Ollama preset
            "", "", "",
            "2",    # OpenAI preset
            "", "", "", "", "", "",
            "2",    # teacher = same as frontier
            "", "", "0.5", "", "", "", "", "",
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


class TestCLI(unittest.TestCase):
    def test_config_command_loads_file_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "rovik.env")
            config.write_env_file(cfg_path, {
                "ROVIK_LOCAL_MODEL": "qwen-test",
                "ROVIK_ESC_KEYNAME": "OPENROUTER_API_KEY",
            })
            buf = io.StringIO()
            with unittest.mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "super-secret-value"}, clear=True), \
                    contextlib.redirect_stdout(buf):
                rc = cli.main(["--config", cfg_path, "config"])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("qwen-test", out)
            self.assertIn("OPENROUTER_API_KEY (env: set)", out)
            self.assertNotIn("super-secret-value", out)

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
