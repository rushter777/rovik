"""Shadow-log joiner — async enrichment of decision records with outcome events.

Reads decisions from data/shadow_log.jsonl (or shadow_log_proxy.jsonl),
outcome events from data/outcome_events.jsonl, joins on decision_id or
(task_text_hash ±5min window), resolves via precedence rules, and writes
enriched copies to data/shadow_log_enriched.jsonl.

The original log files are NEVER rewritten — append-only discipline.
This is a cron / sidecar-tick process, never blocking the request path.

Usage:
    python -m serving.outcome_joiner [--shadow <file>] [--events <file>] [--output <file>] [--watermark-mins 30]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.environ.get("ROVIK_LOG_DIR", os.path.join(REPO_ROOT, "data"))
DEFAULT_SHADOW = os.path.join(LOG_DIR, "shadow_log.jsonl")
DEFAULT_EVENTS = os.path.join(LOG_DIR, "outcome_events.jsonl")
DEFAULT_OUTPUT = os.path.join(LOG_DIR, "shadow_log_enriched.jsonl")
STATE_FILE = os.path.join(LOG_DIR, "joiner_state.json")

# Precedence order: higher number wins. One tool_error beats many retries.
EVENT_PRECEDENCE = {
    "tool_error": 10,
    "user_retry_after_error": 9,
    "assistant_error_msg": 5,
    "human_redo": 3,
    "user_retry": 2,
    "explicit_success": 1,
}

# Confidence threshold: events below this are ignored.
MIN_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string, defaulting naive timestamps to UTC."""
    if not ts_str:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'.
        s = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _ts_diff_minutes(ts_a: str, ts_b: str) -> float | None:
    """Return difference in minutes between two ISO timestamps. None if unparseable."""
    da = _parse_ts(ts_a)
    db = _parse_ts(ts_b)
    if da and db:
        return abs((da - db).total_seconds()) / 60.0
    return None


# ---------------------------------------------------------------------------
# State management (byte-offset checkpoint for resume-ability)
# ---------------------------------------------------------------------------

def load_state(state_path: str = STATE_FILE) -> dict:
    """Load joiner state from disk."""
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"shadow_offset": 0, "events_offset": 0, "last_run": None}


def save_state(state: dict, state_path: str = STATE_FILE):
    """Persist joiner state to disk."""
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Core join logic
# ---------------------------------------------------------------------------

def _resolve_outcome(events: list[dict]) -> str | None:
    """Resolve a set of events to a single outcome using precedence rules.

    Precedence (highest wins):
        tool_error > user_retry_after_error > assistant_error_msg > human_redo > user_retry > explicit_success

    If the winning event is explicit_success and there are no higher-precedence
    events, outcome = "success". Otherwise outcome = the event type name.
    If no events pass confidence threshold, return None.
    """
    # Filter by confidence.
    valid = [e for e in events if e.get("confidence", 0) >= MIN_CONFIDENCE]
    if not valid:
        return None

    # Sort by precedence (descending), then by confidence (descending).
    valid.sort(key=lambda e: (EVENT_PRECEDENCE.get(e["event_type"], 0), e.get("confidence", 0)), reverse=True)
    winner = valid[0]
    etype = winner["event_type"]

    if etype == "explicit_success":
        return "success"
    elif etype in ("tool_error", "assistant_error_msg"):
        return "failed"
    elif etype in ("user_retry", "user_retry_after_error"):
        return "retried"
    elif etype == "human_redo":
        return "human_redo"
    else:
        return None


def _find_events_for_decision(decision_id: str | None, task_text_hash: str | None,
                               decision_ts: str, watermark_mins: int,
                               all_events: list[dict]) -> list[dict]:
    """Find outcome events matching a decision record.

    Resolution order:
        1. Primary: decision_id exact match (within watermark window).
        2. Secondary: task_text_hash ±5min window.
        3. No match → quarantine.
    """
    matched = []

    # Primary: decision_id.
    if decision_id:
        for e in all_events:
            if e.get("decision_id") == decision_id:
                diff = _ts_diff_minutes(decision_ts, e.get("event_ts", ""))
                if diff is not None and diff <= watermark_mins:
                    matched.append(e)

    # Secondary: task_text_hash ±5min.
    if not matched and task_text_hash:
        for e in all_events:
            if e.get("task_text_hash") == task_text_hash:
                diff = _ts_diff_minutes(decision_ts, e.get("event_ts", ""))
                if diff is not None and 5.0 >= diff >= 0:
                    matched.append(e)

    return matched


# ---------------------------------------------------------------------------
# Main joiner loop
# ---------------------------------------------------------------------------

def run_joiner(shadow_path: str = DEFAULT_SHADOW, events_path: str = DEFAULT_EVENTS,
               output_path: str = DEFAULT_OUTPUT, watermark_mins: int = 30,
               state_path: str = STATE_FILE):
    """Run the enrichment pipeline.

    Reads only new lines since last run (byte-offset checkpoint).
    Writes enriched records to a *separate* output file — originals untouched.
    """
    state = load_state(state_path)
    shadow_offset = state.get("shadow_offset", 0)
    events_offset = state.get("events_offset", 0)

    # Load new outcome events since last run.
    new_events = []
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            f.seek(events_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    new_events.append(evt)
                except json.JSONDecodeError:
                    pass
            events_offset = f.tell()
    except FileNotFoundError:
        pass

    # Load decisions to enrich.
    new_decisions = []
    try:
        with open(shadow_path, "r", encoding="utf-8") as f:
            f.seek(shadow_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    new_decisions.append(rec)
                except json.JSONDecodeError:
                    pass
            shadow_offset = f.tell()
    except FileNotFoundError:
        print(f"Shadow log not found at {shadow_path} — nothing to enrich.", flush=True)
        return

    if not new_decisions and not new_events:
        print("Nothing new to process.", flush=True)
        save_state(
            {"shadow_offset": shadow_offset, "events_offset": events_offset,
             "last_run": datetime.now(timezone.utc).isoformat()},
            state_path,
        )
        return

    # Build a lookup from existing enriched records (to avoid duplicates).
    enriched_ids = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("decision_id"):
                        enriched_ids.add(rec["decision_id"])
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass

    # Match events to decisions:
    # 1. Primary: decision_id exact match within watermark window.
    # 2. Secondary: for events without a primary match / decision_id,
    #    (task_text_hash, ±5min window) only when unambiguous (exact 1 match).
    events_by_decision: dict[int, list[dict]] = {i: [] for i in range(len(new_decisions))}
    matched_event_indices: set[int] = set()

    pending_indices = [
        d_idx for d_idx, rec in enumerate(new_decisions)
        if not (rec.get("decision_id") and rec["decision_id"] in enriched_ids)
    ]
    pending_set = set(pending_indices)

    for e_idx, evt in enumerate(new_events):
        edid = evt.get("decision_id")
        if edid:
            for d_idx in pending_indices:
                rec = new_decisions[d_idx]
                if rec.get("decision_id") == edid:
                    diff = _ts_diff_minutes(rec.get("ts", ""), evt.get("event_ts", ""))
                    if diff is not None and diff <= watermark_mins:
                        events_by_decision[d_idx].append(evt)
                        matched_event_indices.add(e_idx)

    for e_idx, evt in enumerate(new_events):
        if e_idx in matched_event_indices or evt.get("decision_id"):
            continue
        ehash = evt.get("task_text_hash")
        if not ehash:
            continue
        candidates = []
        for d_idx, rec in enumerate(new_decisions):
            if rec.get("task_text_hash") == ehash:
                diff = _ts_diff_minutes(rec.get("ts", ""), evt.get("event_ts", ""))
                if diff is not None and 0 <= diff <= 5.0:
                    candidates.append(d_idx)
        if len(candidates) == 1 and candidates[0] in pending_set:
            events_by_decision[candidates[0]].append(evt)
            matched_event_indices.add(e_idx)

    # Enrich decisions with outcome events.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    enriched_count = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for d_idx in pending_indices:
            rec = new_decisions[d_idx]
            did = rec.get("decision_id")
            if did and did in enriched_ids:
                continue  # Already enriched.

            outcome_events = events_by_decision[d_idx]

            # Resolve to a single outcome.
            outcome = _resolve_outcome(outcome_events)
            enriched_rec = dict(rec)
            enriched_rec["execution_outcome"] = outcome
            enriched_rec["_event_count"] = len(outcome_events)
            enriched_rec["_events_matched"] = [e.get("event_type") for e in outcome_events]

            out.write(json.dumps(enriched_rec, ensure_ascii=False) + "\n")
            if did:
                enriched_ids.add(did)
            enriched_count += 1

    # Also write events that have no matching decision (quarantine).
    quarantine_path = os.path.join(os.path.dirname(output_path) or ".", "outcome_unmatched.jsonl")
    unmatched = [e for idx, e in enumerate(new_events) if idx not in matched_event_indices]
    if unmatched:
        with open(quarantine_path, "a", encoding="utf-8") as qf:
            for evt in unmatched:
                qf.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # Persist state.
    save_state(
        {
            "shadow_offset": shadow_offset,
            "events_offset": events_offset,
            "last_run": datetime.now(timezone.utc).isoformat(),
            "enriched_count": enriched_count,
            "quarantined_events": len(unmatched),
        },
        state_path,
    )

    print(f"Enriched {enriched_count} decision(s), quarantined {len(unmatched)} event(s).", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    shadow_path = DEFAULT_SHADOW
    events_path = DEFAULT_EVENTS
    output_path = DEFAULT_OUTPUT
    state_path = STATE_FILE
    watermark_mins = 30

    i = 0
    while i < len(args):
        if args[i] == "--shadow" and i + 1 < len(args):
            shadow_path = args[i + 1]; i += 2
        elif args[i] == "--events" and i + 1 < len(args):
            events_path = args[i + 1]; i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]; i += 2
        elif args[i] == "--state" and i + 1 < len(args):
            state_path = args[i + 1]; i += 2
        elif args[i] == "--watermark-mins" and i + 1 < len(args):
            watermark_mins = int(args[i + 1]); i += 2
        else:
            print(f"Unknown arg: {args[i]}", flush=True)
            return

    run_joiner(shadow_path, events_path, output_path, watermark_mins, state_path)


if __name__ == "__main__":
    main()
