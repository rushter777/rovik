"""Prepare a custom prompt corpus into a labeled train/eval split for Laya RLCD training.

Bridges raw user task histories (.txt, .csv, .jsonl, .json, proxy_log.jsonl,
shadow_log.jsonl) to the exact {state, questions, labels} JSONL format consumed
by pipeline/train_rlcd.py and pipeline/eval_laya.py.

Stdlib-only. Safe to inspect or split without torch/mlx installed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rovik.config import read_secret_value  # noqa: E402
from serving.hook import QUESTION_SET  # noqa: E402

LABEL_FIELDS = (
    "tools_file",
    "tools_terminal",
    "tools_web",
    "needs_verification",
    "q1",
    "q2",
    "q4",
    "q5",
)

TEXT_KEYS = ("state", "task_text", "task", "prompt", "query", "text")
CSV_TEXT_KEYS = TEXT_KEYS + ("message",)

FULL_RUBRIC = """You are labeling a task that a user gave to an AI assistant (which has file, shell, and web tools). Answer all 8 yes/no questions using ONLY the request text. Answer strictly as JSON.

Operational gates:
tools_file: Does this task involve reading or editing files? ("Explain TCP vs UDP" = false. "Rename foo to bar in utils.py" = true.)
tools_terminal: Does this task involve running shell commands or system operations? ("Draft a haiku" = false. "Run pytest and paste output" = true.)
tools_web: Does this task involve looking things up online or fetching web content? ("Delete unused imports in app.py" = false. "Look up latest llama.cpp release" = true.)
needs_verification: Does an answer to this task need double-checking before it is used (high stakes or errors would cost something vs quick low-stakes answer)?

Difficulty decomposition axes:
q1 (multi_step): Would a correct solution require at least two sequential actions where a later action depends on the result of an earlier one? (Two independent asks in one message do NOT count. "Write this function in chat" = false. "Find the config file, change the variable, run tests, fix failures" = true.)
q2 (discovery): Must the assistant first find an unknown (which file, which bug, why it fails, which source is authoritative) rather than execute a procedure the user already named? ("Run pytest and paste output" = false. "Something in CI is red, fix it" = true.)
q4 (design): Must the assistant choose among plausible approaches that could meaningfully affect the result (not merely fill in trivial defaults)? ("Convert this CSV to JSON" = false. "Design an authentication system for this app" = true.)
q5 (exec_verify): Does successful completion require verifying the result by execution, testing, external observation, or comparison against an objective criterion — is a plausible-looking answer alone insufficient? ("Write a regex matching these examples" (regex only requested) = false. "Fix this parser and make all tests pass" = true.)

Respond with ONLY this JSON, no other text:
{"tools_file": true|false, "tools_terminal": true|false, "tools_web": true|false, "needs_verification": true|false, "q1": true|false, "q2": true|false, "q4": true|false, "q5": true|false}

Task text:
"""


def _extract_latest_user_from_messages(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):
                txt = " ".join(
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                if txt.strip():
                    return txt
    return ""


def _extract_text_from_row(row, column: str | None = None) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""
    if column:
        val = row.get(column)
        return val if isinstance(val, str) else ("" if val is None else str(val))
    for key in TEXT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    msg_val = row.get("messages")
    if isinstance(msg_val, list):
        return _extract_latest_user_from_messages(msg_val)
    scalar_msg = row.get("message")
    if isinstance(scalar_msg, str) and scalar_msg.strip():
        return scalar_msg
    return ""


def load_raw_tasks(
    path: str,
    *,
    column: str | None = None,
    min_chars: int = 5,
) -> dict:
    """Load, extract, filter, and deduplicate task prompts from txt/csv/jsonl/json."""
    lower = path.lower()
    raw_items: list = []

    if lower.endswith(".txt"):
        fmt = "txt"
        with open(path, "r", encoding="utf-8") as f:
            raw_items = [line.strip() for line in f if line.strip()]
    elif lower.endswith(".csv"):
        fmt = "csv"
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            chosen_col = column
            if not chosen_col:
                for cand in CSV_TEXT_KEYS:
                    if cand in fieldnames:
                        chosen_col = cand
                        break
                if not chosen_col and fieldnames:
                    chosen_col = fieldnames[0]
            for r in reader:
                raw_items.append(r)
            column = chosen_col
    elif lower.endswith(".jsonl"):
        fmt = "jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    raw_items.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    elif lower.endswith(".json"):
        fmt = "json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            for list_key in ("tasks", "rows", "data", "items"):
                if isinstance(data.get(list_key), list):
                    raw_items = data[list_key]
                    break
            else:
                raw_items = [data]
    else:
        fmt = "txt"
        with open(path, "r", encoding="utf-8") as f:
            raw_items = [line.strip() for line in f if line.strip()]

    dict_items = [r for r in raw_items if isinstance(r, dict)]
    already_labeled = bool(dict_items) and all(
        isinstance(r.get("questions"), dict) and isinstance(r.get("labels"), dict)
        for r in dict_items
    )

    tasks: list[str] = []
    seen: set[str] = set()
    skipped_short = 0
    duplicates_removed = 0

    for item in raw_items:
        text = _extract_text_from_row(item, column=column).strip()
        if len(text) < min_chars:
            skipped_short += 1
            continue
        if text in seen:
            duplicates_removed += 1
            continue
        seen.add(text)
        tasks.append(text)

    return {
        "format": fmt,
        "raw_count": len(raw_items),
        "skipped_short": skipped_short,
        "duplicates_removed": duplicates_removed,
        "already_labeled": already_labeled,
        "tasks": tasks,
    }


def _is_loopback_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def ask_teacher(
    task_text: str,
    *,
    api_url: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.0,
    retries: int = 3,
    timeout_s: int = 120,
) -> dict[str, bool] | None:
    """Query an OpenAI-compatible teacher model for strict-JSON labels across all 8 fields."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": FULL_RUBRIC + task_text}],
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0.0 else 0.8,
        "presence_penalty": 0.0,
        "max_tokens": 120,
    }
    if _is_loopback_url(api_url):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"].strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end < start:
                raise ValueError("no JSON object in teacher response")
            obj = json.loads(raw[start : end + 1])
            vals: dict[str, bool] = {}
            for field in LABEL_FIELDS:
                val = obj.get(field)
                if not isinstance(val, bool):
                    raise ValueError(f"missing or non-bool field: {field}")
                vals[field] = val
            return vals
        except Exception:
            if attempt >= retries - 1:
                return None
            time.sleep(2)
    return None


def build_training_row(task_text: str, raw_labels: dict[str, bool]) -> dict:
    """Convert teacher boolean answers into the {state, questions, labels} training schema."""
    return {
        "state": task_text,
        "questions": QUESTION_SET,
        "labels": {
            "tools_file": int(raw_labels["tools_file"]),
            "tools_terminal": int(raw_labels["tools_terminal"]),
            "tools_web": int(raw_labels["tools_web"]),
            "needs_verification": int(raw_labels["needs_verification"]),
            "difficulty": sum(int(raw_labels[k]) for k in ("q1", "q2", "q4", "q5")),
        },
    }


def run_agreement_gate(
    tasks: list[str],
    *,
    sample_size: int = 100,
    seed: int = 42,
    ask_fn=None,
    report_path: str | None = None,
) -> dict:
    """Run two-pass temperature>0 teacher self-agreement across all 8 rubric fields."""
    if ask_fn is None:
        raise ValueError("ask_fn is required")
    rng = random.Random(seed)
    sample = rng.sample(tasks, min(sample_size, len(tasks))) if tasks else []

    pairs: list[tuple[dict[str, bool], dict[str, bool]]] = []
    skipped = 0
    for task in sample:
        pass1 = ask_fn(task, temperature=0.7)
        pass2 = ask_fn(task, temperature=0.7)
        if pass1 is None or pass2 is None:
            skipped += 1
            continue
        pairs.append((pass1, pass2))

    fields_report: dict[str, dict] = {}
    n_eval = len(pairs)
    for field in LABEL_FIELDS:
        if n_eval == 0:
            fields_report[field] = {"agreement": 0.0, "yes_rate": 0.0, "passed": False}
            continue
        agree = sum(1 for p1, p2 in pairs if p1[field] == p2[field]) / n_eval
        yes_rate = sum(1 for p1, _ in pairs if p1[field]) / n_eval
        fields_report[field] = {
            "agreement": round(agree, 4),
            "yes_rate": round(yes_rate, 4),
            "passed": agree >= 0.85,
        }

    overall_passed = n_eval > 0 and all(v["passed"] for v in fields_report.values())
    report = {
        "sampled": len(sample),
        "evaluated": n_eval,
        "skipped": skipped,
        "threshold": 0.85,
        "overall_passed": overall_passed,
        "fields": fields_report,
    }
    if report_path:
        parent = os.path.dirname(report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def split_labeled_rows(
    rows: list[dict],
    *,
    out_dir: str,
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Deterministically shuffle labeled rows and write train.jsonl + eval.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    if n >= 2 and eval_ratio > 0:
        n_eval = min(n - 1, max(1, round(n * eval_ratio)))
    else:
        n_eval = 0

    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:]

    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    diff_counts = dict(sorted(Counter(r["labels"].get("difficulty", 0) for r in shuffled).items()))
    return {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_path": train_path,
        "eval_path": eval_path,
        "difficulty_counts": diff_counts,
    }


def _task_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def label_and_split(
    tasks: list[str],
    *,
    out_dir: str,
    eval_ratio: float = 0.15,
    seed: int = 42,
    ask_fn=None,
    limit: int | None = None,
    print_fn=print,
) -> dict:
    """Label tasks with resume support in labeled_corpus.jsonl, then write train/eval splits."""
    if ask_fn is None:
        raise ValueError("ask_fn is required")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "labeled_corpus.jsonl")

    all_rows: list[dict] = []
    seen_hashes: set[str] = set()
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(row.get("state"), str) and isinstance(row.get("labels"), dict):
                    h = _task_hash(row["state"])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_rows.append(row)

    skipped_existing = 0
    newly_labeled = 0
    failed = 0
    attempted_new = 0

    with open(checkpoint_path, "a", encoding="utf-8") as out:
        for task in tasks:
            h = _task_hash(task)
            if h in seen_hashes:
                skipped_existing += 1
                continue
            if limit is not None and attempted_new >= limit:
                break
            attempted_new += 1
            vals = ask_fn(task, temperature=0.0)
            if vals is None:
                failed += 1
                continue
            row = build_training_row(task, vals)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            seen_hashes.add(h)
            all_rows.append(row)
            newly_labeled += 1
            if newly_labeled % 10 == 0:
                print_fn(f"  labeled {newly_labeled} new task(s) (total={len(all_rows)})...")

    split_info = split_labeled_rows(all_rows, out_dir=out_dir, eval_ratio=eval_ratio, seed=seed)
    return {
        "total_labeled": len(all_rows),
        "newly_labeled": newly_labeled,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "checkpoint_path": checkpoint_path,
        **split_info,
    }


def _load_prelabeled_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and isinstance(r.get("state"), str) and isinstance(r.get("labels"), dict):
                rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rovik prepare-data",
        description="Inspect, agreement-test, label, and split a prompt corpus for Laya training",
    )
    parser.add_argument("--input", default=os.environ.get("ROVIK_CORPUS_PATH", ""),
                        help="Path to raw tasks (.txt, .csv, .jsonl, .json) or labeled JSONL")
    parser.add_argument("--column", default=None, help="Explicit column/key name containing prompt text")
    parser.add_argument("--min-chars", type=int, default=5, help="Skip tasks shorter than this (default: 5)")
    parser.add_argument("--inspect", action="store_true", help="Inspect file format & counts without calling teacher")
    parser.add_argument("--check-agreement", type=int, default=0, metavar="N",
                        help="Run 2-pass self-agreement test on N tasks before labeling")
    parser.add_argument("--strict-gate", action="store_true",
                        help="Exit non-zero if any field falls below 85%% self-agreement")
    parser.add_argument("--split-only", action="store_true",
                        help="Split an already-labeled JSONL corpus into train/eval without calling teacher")
    parser.add_argument("--out-dir", default=os.environ.get("ROVIK_LOG_DIR") or "data",
                        help="Output directory for labeled_corpus.jsonl, train.jsonl, eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Eval split fraction (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splitting")
    parser.add_argument("--limit", type=int, default=None, help="Max new tasks to label in this run")
    parser.add_argument(
        "--teacher-url",
        default=os.environ.get("ROVIK_TEACHER_URL")
        or os.environ.get("ROVIK_LOCAL_URL")
        or "http://127.0.0.1:8081/v1/chat/completions",
        help="OpenAI-compatible teacher chat completions URL",
    )
    parser.add_argument(
        "--teacher-model",
        default=os.environ.get("ROVIK_TEACHER_MODEL")
        or os.environ.get("ROVIK_LOCAL_MODEL")
        or "local-model",
        help="Teacher model ID",
    )
    parser.add_argument(
        "--teacher-keyname",
        default=os.environ.get("ROVIK_TEACHER_KEYNAME", ""),
        help="Env var name holding API key for remote teacher",
    )
    args = parser.parse_args(argv)

    if not args.input:
        print("Error: --input is required (or set ROVIK_CORPUS_PATH in rovik.env).", file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f"Error: corpus file not found: {args.input}", file=sys.stderr)
        return 2

    if args.inspect:
        info = load_raw_tasks(args.input, column=args.column, min_chars=args.min_chars)
        print(f"File:               {args.input}")
        print(f"Format:             {info['format']}")
        print(f"Raw records:        {info['raw_count']}")
        print(f"Unique valid tasks: {len(info['tasks'])}")
        print(f"Skipped (<{args.min_chars} chars): {info['skipped_short']}")
        print(f"Duplicates removed: {info['duplicates_removed']}")
        print(f"Already labeled:    {info['already_labeled']}")
        if info["tasks"]:
            print("Preview (first 3):")
            for idx, task in enumerate(info["tasks"][:3], 1):
                snippet = task.replace("\n", " ")[:90]
                print(f"  {idx}. {snippet}")
        return 0

    if args.split_only:
        rows = _load_prelabeled_rows(args.input)
        if not rows:
            print(f"Error: no pre-labeled {{state, labels}} rows found in {args.input}", file=sys.stderr)
            return 2
        split_info = split_labeled_rows(rows, out_dir=args.out_dir, eval_ratio=args.eval_ratio, seed=args.seed)
        print(f"Split {len(rows)} rows -> {split_info['train_count']} train ({split_info['train_path']}), "
              f"{split_info['eval_count']} eval ({split_info['eval_path']})")
        return 0

    info = load_raw_tasks(args.input, column=args.column, min_chars=args.min_chars)
    tasks = info["tasks"]
    if not tasks:
        print("Error: no valid tasks found after filtering.", file=sys.stderr)
        return 2

    api_key = read_secret_value(args.teacher_keyname) if args.teacher_keyname else ""

    def _ask(task_text: str, temperature: float = 0.0) -> dict[str, bool] | None:
        return ask_teacher(
            task_text,
            api_url=args.teacher_url,
            model=args.teacher_model,
            api_key=api_key,
            temperature=temperature,
        )

    if args.check_agreement > 0:
        report_path = os.path.join(args.out_dir, "agreement_report.json")
        rep = run_agreement_gate(
            tasks,
            sample_size=args.check_agreement,
            seed=args.seed,
            ask_fn=_ask,
            report_path=report_path,
        )
        print(f"\nAgreement Gate ({rep['evaluated']} evaluated, {rep['skipped']} skipped):")
        for field, stats in rep["fields"].items():
            verdict = "PASS" if stats["passed"] else "FAIL"
            print(f"  {field:<20s} agreement={stats['agreement']:.1%} yes={stats['yes_rate']:.0%} [{verdict}]")
        print(f"Report saved to {report_path}")
        if args.strict_gate and not rep["overall_passed"]:
            print("Strict gate failed: one or more fields scored <85% self-agreement.", file=sys.stderr)
            return 1

    res = label_and_split(
        tasks,
        out_dir=args.out_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        ask_fn=_ask,
        limit=args.limit,
    )
    print(f"\nLabeled corpus: {res['total_labeled']} total "
          f"({res['newly_labeled']} new, {res['skipped_existing']} resumed, {res['failed']} failed)")
    print(f"Train split:    {res['train_count']} rows -> {res['train_path']}")
    print(f"Eval split:     {res['eval_count']} rows -> {res['eval_path']}")
    print(f"Next step:      rovik train --data {res['train_path']} --output-dir checkpoints/laya-ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
