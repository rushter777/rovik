#!/usr/bin/env python3
"""Agreement test for the 5-question difficulty decomposition (Sept 2026).

Merged question set (from 4-model comparison: Helix Max/Gemini/ChatGPT/Grok):
  q1 multi_step   — >=2 sequential actions, later depends on earlier result
  q2 discovery    — must find an unknown (which file/bug/why) vs execute named procedure
  q3 done_open    — >1 reasonable interpretation of "done" (format/scope/acceptance unstated)
  q4 design       — must choose among plausible approaches (design vs execute)
  q5 exec_verify  — must run/test/observe to prove it works (plausible answer insufficient)

Protocol: sample N task texts from a corpus JSONL, ask the teacher model
(all 5 questions twice per task, temp 0.7, two independent draws) and report
per-question self-agreement. Gate: >=85% agreement to earn a node.
Teacher: any OpenAI-compatible local server (ROVIK_TEACHER_URL, default
http://127.0.0.1:8081/v1/chat/completions; model name via ROVIK_TEACHER_MODEL).
Stdlib + urllib only (no deps).
"""

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

API = os.environ.get("ROVIK_TEACHER_URL", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("ROVIK_TEACHER_MODEL", "local-model")
OUT = os.environ.get("ROVIK_AGREEMENT_OUT", "data/agreement_test.jsonl")

RUBRIC = """You are labeling a task that a user gave to an AI assistant (which has file, shell, and web tools). Answer 5 yes/no questions using ONLY the request text. Two careful readers must be able to agree from the text alone. Answer strictly as JSON.

q1 multi_step: Would a correct solution require at least two sequential actions where a later action depends on the result of an earlier one? (Two independent asks in one message do NOT count. "Write this function in chat" = false. "Find the config file, change the variable, run tests, fix failures" = true.)

q2 discovery: Must the assistant first find an unknown (which file, which bug, why it fails, which source is authoritative) rather than execute a procedure the user already named? ("Run pytest and paste output" = false. "Something in CI is red, fix it" = true.)

q3 done_open: Is there more than one reasonable interpretation of "done" because format, constraints, scope, edge cases, or acceptance criteria are left unstated? ("Rename foo to bar in these 3 files and keep tests green" = false. "Refactor this module" = true. This is NOT "is it hard".)

q4 design: Must the assistant choose among plausible approaches that could meaningfully affect the result (not merely fill in trivial defaults)? ("Convert this CSV to JSON" = false. "Design an authentication system for this app" = true.)

q5 exec_verify: Does successful completion require verifying the result by execution, testing, external observation, or comparison against an objective criterion — is a plausible-looking answer alone insufficient? ("Write a regex matching these examples" (regex only requested) = false. "Fix this parser and make all tests pass" = true.)

Respond with ONLY this JSON, no other text:
{"q1": true|false, "q2": true|false, "q3": true|false, "q4": true|false, "q5": true|false}

Task text:
"""


def ask(task_text, retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": RUBRIC + task_text}],
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "max_tokens": 60,
    }
    if (urllib.parse.urlparse(API).hostname or "").lower() in ("127.0.0.1", "localhost", "::1"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    key_name = os.environ.get("ROVIK_TEACHER_KEYNAME", "").strip()
    api_key = os.environ.get(key_name, "").strip() if key_name else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            raw = resp["choices"][0]["message"]["content"].strip()
            # find the JSON object in the response
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1:
                raise ValueError(f"no JSON in: {raw[:80]}")
            obj = json.loads(raw[start:end + 1])
            vals = {}
            for q in ("q1", "q2", "q3", "q4", "q5"):
                if not isinstance(obj.get(q), bool):
                    raise ValueError(f"missing/bad {q}")
                vals[q] = obj[q]
            return vals
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(2)


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/synthetic_corpus.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    sample = random.sample(rows, min(n_sample, len(rows)))
    print(f"Asking {len(sample)} tasks x 2 passes x 5 questions on the MoE...", flush=True)

    results = []
    for i, r in enumerate(sample):
        text = r.get("state") or r.get("task_text") or ""
        if not text:
            continue
        a1 = ask(text)
        a2 = ask(text)
        if a1 is None or a2 is None:
            print(f"  [{i}] parse failure on both passes, skipping", flush=True)
            continue
        results.append({"task": text[:200], "pass1": a1, "pass2": a2})
        if (i + 1) % 5 == 0:
            agree = sum(1 for x in results for q in ("q1", "q2", "q3", "q4", "q5")
                        if x["pass1"][q] == x["pass2"][q])
            total = len(results) * 5
            print(f"  {i+1}/{len(sample)} done | running agreement {agree}/{total} = {agree/total:.1%}", flush=True)
        with open(OUT, "w") as f:  # checkpoint each iteration
            for x in results:
                f.write(json.dumps(x) + "\n")

    print(f"\n=== FINAL: {len(results)} tasks with both passes ===")
    labels = {"q1": "multi_step", "q2": "discovery", "q3": "done_open",
              "q4": "design", "q5": "exec_verify"}
    for q, name in labels.items():
        agree = sum(1 for x in results if x["pass1"][q] == x["pass2"][q])
        n = len(results)
        yes_rate = sum(1 for x in results if x["pass1"][q]) / n
        verdict = "PASS" if agree / n >= 0.85 else "FAIL"
        print(f"  {name:12s} agreement {agree}/{n} = {agree/n:.1%}  yes-rate {yes_rate:.0%}  [{verdict} >=85%]")


if __name__ == "__main__":
    main()
