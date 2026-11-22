"""Stdlib-only unit tests for shadow.py, outcome_extract.py, and outcome_joiner.py."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from serving import outcome_extract, outcome_joiner, shadow  # noqa: E402


class TestShadow(unittest.TestCase):
    def test_extract_p_hard_supports_both_zero_keys(self):
        rec_idx = {"raw": {"difficulty": {"probabilities": {"0": 0.25, "1": 0.75}}}}
        rec_lbl = {"raw": {"difficulty": {"probabilities": {"0-none": 0.4, "1-one demand": 0.6}}}}
        self.assertAlmostEqual(shadow._extract_p_hard(rec_idx), 0.75)
        self.assertAlmostEqual(shadow._extract_p_hard(rec_lbl), 0.6)

    def test_process_record_fields(self):
        entry = shadow._process_record({
            "task_text": "check status",
            "source": "laya",
            "decision_id": "abc123def456",
            "task_text_hash": "0123456789abcdef",
            "routed_to": "local",
            "raw": {
                "tools_file": {"noul": 0.1},
                "tools_terminal": {"noul": 0.8},
                "tools_web": {"noul": 0.05},
                "needs_verification": {"noul": 0.2},
                "difficulty": {"probabilities": {"0": 0.9}},
            },
        })
        self.assertAlmostEqual(entry["tools_terminal"], 0.8)
        self.assertAlmostEqual(entry["p_hard"], 0.1)
        self.assertEqual(entry["decision_id"], "abc123def456")
        self.assertEqual(entry["routed_to"], "local")


class TestOutcomeExtract(unittest.TestCase):
    def test_auto_extracts_decision_id_hash_errors_and_retries(self):
        task_text = "Find why the test suite fails and fix it"
        expected_hash = hashlib.sha256(task_text.encode()).hexdigest()[:16]
        msgs = [
            {"role": "system", "content": "ROUTING DECISION ID: aabbcc112233 — include this id."},
            {"role": "user", "content": task_text, "ts": "2026-11-22T10:00:00Z"},
            {"role": "tool_result", "tool_name": "bash", "content": "exit code 1: AssertionError", "ts": "2026-11-22T10:00:05Z"},
            {"role": "assistant", "content": "error: tests failed on line 10.", "ts": "2026-11-22T10:00:08Z"},
            {"role": "user", "content": "no, try again with the right config", "ts": "2026-11-22T10:00:30Z"},
            {"role": "assistant", "content": "Fixed and passing now.", "ts": "2026-11-22T10:00:40Z"},
            {"role": "user", "content": "perfect, thanks!", "ts": "2026-11-22T10:00:50Z"},
        ]
        events = outcome_extract.extract_events_from_transcript(msgs)
        types_found = [e["event_type"] for e in events]
        self.assertIn("tool_error", types_found)
        self.assertIn("assistant_error_msg", types_found)
        self.assertIn("user_retry_after_error", types_found)
        self.assertIn("explicit_success", types_found)

        retry_evt = next(e for e in events if e["event_type"] == "user_retry_after_error")
        self.assertEqual(retry_evt["decision_id"], "aabbcc112233")
        self.assertEqual(retry_evt["task_text_hash"], expected_hash)

    def test_retry_compares_user_turns_and_scopes_prior_errors(self):
        # Turn 1 fails -> assistant replies -> user asks a NEW clean task -> assistant replies ->
        # user repeats the second user turn ("try again"). Must be user_retry (not user_retry_after_error)
        # because the error happened before the second task, not between turns 3 and 5.
        msgs = [
            {"role": "system", "content": "ROUTING DECISION ID: dec111111111"},
            {"role": "user", "content": "Run build script", "ts": "2026-11-22T10:00:00Z"},
            {"role": "tool_result", "content": "error: build failed", "ts": "2026-11-22T10:00:02Z"},
            {"role": "assistant", "content": "Build failed.", "ts": "2026-11-22T10:00:03Z"},
            {"role": "system", "content": "ROUTING DECISION ID: dec222222222"},
            {"role": "user", "content": "Explain the difference between TCP and UDP in plain concise words", "ts": "2026-11-22T10:01:00Z"},
            {"role": "assistant", "content": "Here is an unrelated long explanation of sockets.", "ts": "2026-11-22T10:01:05Z"},
            {"role": "user", "content": "Explain the difference between TCP and UDP in plain concise words please", "ts": "2026-11-22T10:01:15Z"},
        ]
        events = outcome_extract.extract_events_from_transcript(msgs)
        retries = [e for e in events if e["event_type"] in ("user_retry", "user_retry_after_error")]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["event_type"], "user_retry")
        self.assertEqual(retries[0]["decision_id"], "dec222222222")

    def test_load_hermes_log_format(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".log", delete=False) as tf:
            tf.write(
                'ts=2026-11-22T10:00:00Z event=user_message payload={"content":"check disk","decision_id":"d99"}\n'
            )
            tf.write(
                'ts=2026-11-22T10:00:02Z event=tool_result payload={"tool_name":"df","error":"permission denied"}\n'
            )
            log_path = tf.name
        try:
            msgs = outcome_extract.load_transcript(log_path)
            self.assertEqual(len(msgs), 2)
            events = outcome_extract.extract_events_from_transcript(msgs)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "tool_error")
            self.assertEqual(events[0]["decision_id"], "d99")
        finally:
            os.unlink(log_path)


class TestOutcomeJoiner(unittest.TestCase):
    def test_resolve_outcome_treats_assistant_error_msg_as_failed(self):
        events = [
            {"event_type": "assistant_error_msg", "confidence": 0.7},
            {"event_type": "explicit_success", "confidence": 0.9},
        ]
        self.assertEqual(outcome_joiner._resolve_outcome(events), "failed")

    def test_parse_ts_handles_naive_and_aware_timestamps(self):
        diff = outcome_joiner._ts_diff_minutes("2026-11-22T10:00:00", "2026-11-22T10:03:00Z")
        self.assertAlmostEqual(diff, 3.0)

    def test_joiner_primary_secondary_ambiguity_quarantine_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shadow_path = os.path.join(tmpdir, "shadow_log.jsonl")
            events_path = os.path.join(tmpdir, "outcome_events.jsonl")
            output_path = os.path.join(tmpdir, "shadow_log_enriched.jsonl")
            state_path = os.path.join(tmpdir, "joiner_state.json")
            quarantine_path = os.path.join(tmpdir, "outcome_unmatched.jsonl")

            decisions = [
                {"ts": "2026-11-22T10:00:00Z", "decision_id": "d_primary", "task_text_hash": "h_primary"},
                {"ts": "2026-11-22T10:01:00Z", "decision_id": "d_sec_ok", "task_text_hash": "h_unique"},
                {"ts": "2026-11-22T10:02:00Z", "decision_id": "d_ambig_1", "task_text_hash": "h_ambig"},
                {"ts": "2026-11-22T10:03:00Z", "decision_id": "d_ambig_2", "task_text_hash": "h_ambig"},
            ]
            events = [
                # Primary match via decision_id (naive ts vs aware decision ts)
                {"event_ts": "2026-11-22T10:00:30", "decision_id": "d_primary", "event_type": "tool_error", "confidence": 0.95},
                # Secondary unique match without decision_id -> NOT quarantined
                {"event_ts": "2026-11-22T10:02:00Z", "decision_id": None, "task_text_hash": "h_unique", "event_type": "explicit_success", "confidence": 0.9},
                # Ambiguous secondary match (matches d_ambig_1 and d_ambig_2) -> quarantined
                {"event_ts": "2026-11-22T10:04:00Z", "decision_id": None, "task_text_hash": "h_ambig", "event_type": "user_retry", "confidence": 0.8},
                # Unmatched event WITH a decision_id -> MUST be quarantined
                {"event_ts": "2026-11-22T10:05:00Z", "decision_id": "d_nonexistent", "task_text_hash": "h_primary", "event_type": "tool_error", "confidence": 0.95},
            ]
            with open(shadow_path, "w", encoding="utf-8") as f:
                for d in decisions:
                    f.write(json.dumps(d) + "\n")
            with open(events_path, "w", encoding="utf-8") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")

            outcome_joiner.run_joiner(shadow_path, events_path, output_path, watermark_mins=30, state_path=state_path)

            with open(output_path, encoding="utf-8") as f:
                enriched = [json.loads(ln) for ln in f]
            by_id = {r["decision_id"]: r for r in enriched}
            self.assertEqual(len(enriched), 4)
            self.assertEqual(by_id["d_primary"]["execution_outcome"], "failed")
            self.assertEqual(by_id["d_sec_ok"]["execution_outcome"], "success")
            self.assertIsNone(by_id["d_ambig_1"]["execution_outcome"])
            self.assertIsNone(by_id["d_ambig_2"]["execution_outcome"])

            with open(quarantine_path, encoding="utf-8") as f:
                quarantined = [json.loads(ln) for ln in f]
            self.assertEqual(len(quarantined), 2)
            q_dids = {q.get("decision_id") for q in quarantined}
            q_hashes = {q.get("task_text_hash") for q in quarantined}
            self.assertIn("d_nonexistent", q_dids)
            self.assertIn("h_ambig", q_hashes)

            # Second run with no new lines should not duplicate enriched records.
            outcome_joiner.run_joiner(shadow_path, events_path, output_path, watermark_mins=30, state_path=state_path)
            with open(output_path, encoding="utf-8") as f:
                enriched_after = [json.loads(ln) for ln in f]
            self.assertEqual(len(enriched_after), 4)


if __name__ == "__main__":
    unittest.main()
