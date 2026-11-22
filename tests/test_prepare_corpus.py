"""Stdlib-only unit tests for pipeline/prepare_corpus.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline import prepare_corpus  # noqa: E402
from serving import hook  # noqa: E402


class TestLoadRawTasks(unittest.TestCase):
    def test_txt_deduplication_and_min_chars(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
            tf.write("hi\n\nExplain TCP vs UDP\nExplain TCP vs UDP\nFix the failing unit test\n")
            path = tf.name
        try:
            res = prepare_corpus.load_raw_tasks(path, min_chars=5)
            self.assertEqual(res["format"], "txt")
            self.assertEqual(res["skipped_short"], 1)
            self.assertEqual(res["duplicates_removed"], 1)
            self.assertEqual(res["tasks"], ["Explain TCP vs UDP", "Fix the failing unit test"])
        finally:
            os.unlink(path)

    def test_csv_auto_and_explicit_column(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", delete=False) as tf:
            tf.write("id,prompt,notes\n1,Check disk usage,ok\n2,Summarize log file,note\n")
            path = tf.name
        try:
            auto = prepare_corpus.load_raw_tasks(path)
            self.assertEqual(auto["tasks"], ["Check disk usage", "Summarize log file"])
            explicit = prepare_corpus.load_raw_tasks(path, column="notes", min_chars=2)
            self.assertEqual(explicit["tasks"], ["ok", "note"])
        finally:
            os.unlink(path)

    def test_jsonl_formats_proxy_shadow_messages_and_json_array(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tf:
            tf.write(json.dumps({"task": "From proxy log"}) + "\n")
            tf.write(json.dumps({"task_text": "From shadow log"}) + "\n")
            tf.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": "From messages list"}]},
                ]
            }) + "\n")
            jsonl_path = tf.name

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as jf:
            json.dump(["Task alpha", {"prompt": "Task beta"}], jf)
            json_path = jf.name
        try:
            r1 = prepare_corpus.load_raw_tasks(jsonl_path)
            self.assertEqual(r1["tasks"], ["From proxy log", "From shadow log", "From messages list"])
            r2 = prepare_corpus.load_raw_tasks(json_path)
            self.assertEqual(r2["tasks"], ["Task alpha", "Task beta"])
        finally:
            os.unlink(jsonl_path)
            os.unlink(json_path)

    def test_inspect_cli_runs_without_http(self):
        syn = os.path.join(_PROJECT_ROOT, "data", "synthetic", "synthetic_corpus.jsonl")
        buf = io.StringIO()
        with unittest.mock.patch("urllib.request.urlopen") as m_open, contextlib.redirect_stdout(buf):
            rc = prepare_corpus.main(["--input", syn, "--inspect"])
        self.assertEqual(rc, 0)
        m_open.assert_not_called()
        out = buf.getvalue()
        self.assertIn("Unique valid tasks: 40", out)
        self.assertIn("Already labeled:    True", out)


class _FakeHTTPResponse:
    def __init__(self, body_dict):
        self._raw = json.dumps(body_dict).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestTeacherAndAgreement(unittest.TestCase):
    def test_ask_teacher_loopback_vs_remote_and_auth(self):
        valid_reply = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "tools_file": True,
                        "tools_terminal": False,
                        "tools_web": False,
                        "needs_verification": True,
                        "q1": True,
                        "q2": False,
                        "q4": False,
                        "q5": True,
                    })
                }
            }]
        }
        seen_requests = []

        def fake_urlopen(req, timeout=120):
            seen_requests.append((req.full_url, dict(req.header_items()), json.loads(req.data.decode("utf-8"))))
            return _FakeHTTPResponse(valid_reply)

        with unittest.mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res_local = prepare_corpus.ask_teacher(
                "Edit foo.py",
                api_url="http://127.0.0.1:8082/v1/chat/completions",
                model="local-35b",
                retries=1,
            )
            res_remote = prepare_corpus.ask_teacher(
                "Edit foo.py",
                api_url="https://openrouter.ai/api/v1/chat/completions",
                model="z-ai/glm-5.3",
                api_key="secret-k",
                retries=1,
            )
            res_agree = prepare_corpus.ask_teacher(
                "Edit foo.py",
                api_url="http://127.0.0.1:8082/v1/chat/completions",
                model="local-35b",
                temperature=0.7,
                retries=1,
            )

        self.assertTrue(res_local["tools_file"])
        self.assertTrue(res_remote["q5"])
        self.assertTrue(res_agree["tools_file"])
        _, local_hdrs, local_body = seen_requests[0]
        _, remote_hdrs, remote_body = seen_requests[1]
        _, _, agree_body = seen_requests[2]
        self.assertIn("chat_template_kwargs", local_body)
        self.assertEqual(local_body["presence_penalty"], 0.0)
        self.assertEqual(local_body["top_p"], 1.0)
        self.assertNotIn("Authorization", local_hdrs)
        self.assertNotIn("chat_template_kwargs", remote_body)
        self.assertEqual(remote_body["presence_penalty"], 0.0)
        self.assertEqual(remote_hdrs.get("Authorization"), "Bearer secret-k")
        self.assertEqual(agree_body["temperature"], 0.7)
        self.assertEqual(agree_body["top_p"], 0.8)
        self.assertEqual(agree_body["presence_penalty"], 0.0)

    def test_legacy_pipeline_scripts_override_presence_penalty_and_gate_chat_kwargs(self):
        from pipeline import agreement_test, label_corpus, tier_agreement_test

        reply = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "q1": True, "q2": False, "q3": False, "q4": False, "q5": True,
                        "local_ok": True,
                    })
                }
            }]
        }
        calls = []

        def fake_urlopen(req, timeout=120):
            calls.append((req.full_url, dict(req.header_items()), json.loads(req.data.decode("utf-8"))))
            return _FakeHTTPResponse(reply)

        with unittest.mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            for mod in (agreement_test, tier_agreement_test, label_corpus):
                with unittest.mock.patch.object(mod, "API", "http://127.0.0.1:8082/v1/chat/completions"):
                    mod.ask("Check disk usage", retries=1)
                with unittest.mock.patch.object(mod, "API", "https://openrouter.ai/api/v1/chat/completions"), \
                        unittest.mock.patch.dict(os.environ, {"ROVIK_TEACHER_KEYNAME": "TK", "TK": "k123"}, clear=False):
                    mod.ask("Check disk usage", retries=1)

        self.assertEqual(len(calls), 6)
        for i in range(0, 6, 2):
            _, local_hdrs, local_body = calls[i]
            _, remote_hdrs, remote_body = calls[i + 1]
            self.assertEqual(local_body["presence_penalty"], 0.0)
            self.assertEqual(remote_body["presence_penalty"], 0.0)
            self.assertIn("chat_template_kwargs", local_body)
            self.assertNotIn("chat_template_kwargs", remote_body)
            self.assertNotIn("Authorization", local_hdrs)
            self.assertEqual(remote_hdrs.get("Authorization"), "Bearer k123")

        # agreement scripts keep stochastic temp=0.7/top_p=0.8; label_corpus uses deterministic 0.0/1.0
        self.assertEqual(calls[0][2]["temperature"], 0.7)
        self.assertEqual(calls[0][2]["top_p"], 0.8)
        self.assertEqual(calls[2][2]["temperature"], 0.7)
        self.assertEqual(calls[2][2]["top_p"], 0.8)
        self.assertEqual(calls[4][2]["temperature"], 0.0)
        self.assertEqual(calls[4][2]["top_p"], 1.0)

    def test_run_agreement_gate_pass_and_fail_and_report(self):
        tasks = [f"Task number {i} for testing" for i in range(6)]
        all_true = {k: True for k in prepare_corpus.LABEL_FIELDS}

        def always_agree(_t, temperature=0.7):
            return dict(all_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            rep_path = os.path.join(tmpdir, "agreement_report.json")
            rep = prepare_corpus.run_agreement_gate(
                tasks, sample_size=4, seed=42, ask_fn=always_agree, report_path=rep_path
            )
            self.assertTrue(rep["overall_passed"])
            self.assertEqual(rep["evaluated"], 4)
            self.assertTrue(os.path.isfile(rep_path))

        flip = {"n": 0}

        def flip_q4(_t, temperature=0.7):
            flip["n"] += 1
            d = dict(all_true)
            d["q4"] = (flip["n"] % 2 == 0)
            return d

        rep_fail = prepare_corpus.run_agreement_gate(tasks, sample_size=4, seed=42, ask_fn=flip_q4)
        self.assertFalse(rep_fail["overall_passed"])
        self.assertFalse(rep_fail["fields"]["q4"]["passed"])
        self.assertEqual(rep_fail["fields"]["q4"]["agreement"], 0.0)


class TestLabelAndSplit(unittest.TestCase):
    def test_label_and_split_schema_resume_and_determinism(self):
        long_task = "Diagnose why the integration test suite fails after refactoring the parser module. " * 5
        self.assertGreater(len(long_task), 250)
        tasks = [long_task] + [f"Short valid task number {i}" for i in range(9)]
        calls: list[str] = []

        def fake_ask(task_text, temperature=0.0):
            calls.append(task_text)
            is_long = task_text == long_task
            return {
                "tools_file": is_long,
                "tools_terminal": is_long,
                "tools_web": False,
                "needs_verification": is_long,
                "q1": is_long,
                "q2": is_long,
                "q4": False,
                "q5": is_long,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            res1 = prepare_corpus.label_and_split(
                tasks, out_dir=tmpdir, eval_ratio=0.2, seed=7, ask_fn=fake_ask, print_fn=lambda _s: None
            )
            self.assertEqual(len(calls), 10)
            self.assertEqual(res1["total_labeled"], 10)
            self.assertEqual(res1["train_count"], 8)
            self.assertEqual(res1["eval_count"], 2)

            with open(res1["train_path"], "r", encoding="utf-8") as f:
                train_rows = [json.loads(ln) for ln in f]
            with open(res1["eval_path"], "r", encoding="utf-8") as f:
                eval_rows = [json.loads(ln) for ln in f]
            all_split_rows = train_rows + eval_rows
            states = {r["state"] for r in all_split_rows}
            self.assertEqual(states, set(tasks))
            self.assertEqual(len(states), 10)

            long_row = next(r for r in all_split_rows if r["state"] == long_task)
            self.assertEqual(long_row["questions"], hook.QUESTION_SET)
            self.assertEqual(long_row["labels"], {
                "tools_file": 1,
                "tools_terminal": 1,
                "tools_web": 0,
                "needs_verification": 1,
                "difficulty": 3,
            })

            # Re-run: all 10 should resume from checkpoint with zero new teacher calls.
            calls.clear()
            res2 = prepare_corpus.label_and_split(
                tasks, out_dir=tmpdir, eval_ratio=0.2, seed=7, ask_fn=fake_ask, print_fn=lambda _s: None
            )
            self.assertEqual(len(calls), 0)
            self.assertEqual(res2["skipped_existing"], 10)
            self.assertEqual(res2["newly_labeled"], 0)
            with open(res2["eval_path"], "r", encoding="utf-8") as f:
                eval_rows_2 = [json.loads(ln) for ln in f]
            self.assertEqual(eval_rows_2, eval_rows)


if __name__ == "__main__":
    unittest.main()
