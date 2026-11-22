# Rovik — Design & Methodology (*The Router That Earns Its Keep*)

**A practical pipeline for fine-tuning a small decision model (Laya) on your own tasks — including what failed, and why.**

*Technical write-up, September 2026. For setup, CLI usage, and bringing your own data, see [README.md](README.md).*

---

## Contents

- [1. The problem](#1-the-problem)
- [2. The method (the part worth stealing)](#2-the-method-the-part-worth-stealing)
  - [2.1 Decompose subjective → objective](#21-decompose-subjective--objective)
  - [2.2 Gate on teacher self-agreement — before labeling anything](#22-gate-on-teacher-self-agreement--before-labeling-anything)
  - [2.3 Train, then eval against the dumb strategy](#23-train-then-eval-against-the-dumb-strategy)
  - [2.4 When you can't beat the average, sweep for precision](#24-when-you-cant-beat-the-average-sweep-for-precision)
  - [2.5 Ship advisory-first, retire honestly](#25-ship-advisory-first-retire-honestly)
- [3. The scoreboard](#3-the-scoreboard)
- [4. Serving architecture, privacy & outcome ledger](#4-serving-architecture-privacy--outcome-ledger)
  - [4.0 Privacy and data handling](#40-privacy-and-data-handling)
  - [4.1 The router (`serving/proxy.py`) — and how to see who answered](#41-the-router-servingproxypy--and-how-to-see-who-answered)
  - [4.2 The savings dashboard](#42-the-savings-dashboard)
  - [4.3 The outcome ledger — closing the loop asynchronously](#43-the-outcome-ledger--closing-the-loop-asynchronously)
- [5. Status, open problems & troubleshooting](#5-status-open-problems--troubleshooting)
- [6. The four-model decomposition experiment](#6-the-four-model-decomposition-experiment)
- [7. Acknowledging the lineage](#7-acknowledging-the-lineage)

---

## 1. The problem

Big LLMs are expensive and slow for one job they're bad at anyway: **fast, consistent yes/no judgments.** "Does this task need terminal access?" "Is this answer high-stakes enough to double-check?" A 400M-parameter classifier can answer those in ~40ms with better consistency than a 70B chat model — if it's trained on the right data.

This project built that classifier for a real workload: ~1,500 actual user tasks given to an AI assistant (file work, shell commands, web lookups, debugging). The stack: **Laya** (Apache-2.0 open decision model, 322–421M params) fine-tuned locally via RLCD, served behind a FastAPI sidecar, consulted by a stdlib-only hook and an OpenAI-compatible routing proxy with a hard fail-safe. Everything runs on a single Mac. No cloud required for judgment or local responses.

The model that did double duty in the original setup — teacher that labeled the corpus, and local tier that answers routine prompts — was **Qwen3.6-35B-A3B** (MoE, ~35B total / ~3B active, 4-bit quant), served locally via `llama.cpp`. That class of model is the practical floor for teacher work: strong enough at strict-JSON rubric labeling to hit 92–96% self-agreement, small enough to run 1,500 labeling calls for free. The escalation target, by contrast, is deliberately unspecified — it's whatever frontier model you'd rather pay for only when needed. Smaller local answering models (e.g. 9B–14B) work with the same router by lowering the `LAYA_THRESH_HARD` escalation cutoff.

---

## 2. The method (the part worth stealing)

The single most important lesson: **subjective questions are untrainable; objective questions are easy.** Everything else in this document is a consequence of that.

### 2.1 Decompose subjective → objective

"Is this task difficult?" is an opinion — two readers disagree, the training labels disagree with themselves, and the model learns noise. Instead, decompose into text-visible facts:

1. **multi_step** — does the task require ≥2 sequential actions where later ones depend on earlier results?
2. **discovery** — must the assistant first find an unknown (which file, which bug), rather than execute a named procedure?
3. **design** — must it choose among plausible approaches?
4. **exec_verify** — does "done" require execution/testing to prove it works?

(This exact set was derived independently by four different AI models asked blind — see [§6](#6-the-four-model-decomposition-experiment).)

A difficulty score is then the **count of yes answers (0–4)**, not a majority vote. Ordinal counts beat named buckets by a wide margin in our experiments: an ordinal "rate 1–5" formulation scored 71% where the equivalent "trivial/moderate/complex" bucket question scored 31%.

### 2.2 Gate on teacher self-agreement — before labeling anything

You will label your corpus with a bigger model (a "teacher"). Before spending hours on labeling, run this test: **ask the teacher the same question twice on ~100 real tasks (`temperature > 0`) and measure agreement.**

- `≥85%` self-agreement → the question is labelable
- `<85%` → the question is an opinion, not a fact — cut it or reformulate

In our run: `multi_step` 96%, `design` 96%, `exec_verify` 93%, `discovery` 92% — all passed. "Is there more than one reasonable interpretation of done?" (`done_open`) came in at **79% — cut**, per the rule set before the test. No renegotiating the bar after the numbers come in.

This test costs an afternoon and saved us from training on noise twice.

### 2.3 Train, then eval against the dumb strategy

RLCD fine-tuning (proper scoring rules + policy gradient, [pipeline/train_rlcd.py](pipeline/train_rlcd.py)) worked cleanly on ~1,200 rows on a single machine (Apple Silicon MPS, ~88 min). The trainer is the easy part.

The eval bar matters more: **the baseline is "always answer the majority class."** For a difficulty distribution that's 77% routine, a model must beat 77% — not "random chance," not "the leaderboard," the *majority class*. A router question is only useful if it beats the strategy that costs zero compute.

### 2.4 When you can't beat the average, sweep for precision

Exact prediction tied the baseline (78.0% vs 77.6%) — because most tasks genuinely are easy, and the hard ones are rare. The fix: stop asking for exact scores and sweep the **precision/coverage curve** ([pipeline/sweep.py](pipeline/sweep.py)). Flag a task as hard only when `P(score >= 1) >= 0.8`:

| Cutoff | Coverage | Precision |
|---|---|---|
| `P >= 0.8` | 6.3% | **81.2%** |
| `P >= 0.7` | 10.6% | 77.8% |

Against a 22% base rate, that's 3.7× the dumb strategy on flagged tasks. **An imprecise question became a deployable high-precision flag** — which is the operationally useful shape anyway ("which tasks deserve escalation?").

### 2.5 Ship advisory-first, retire honestly

- The flag initially shipped **advisory-only**: logged, shadow-recorded, nothing gated on it until live traffic verified calibration.
- Questions that never beat their baseline get **retired with evidence**, not nursed along. Our "tier" question (which model should handle this task — local vs cloud) failed every formulation: 3-class 29%, binary 59.6% (vs 64.7% baseline), teacher self-agreement 81% ([pipeline/tier_agreement_test.py](pipeline/tier_agreement_test.py)), and a post-training probability head that output a constant. It became a **rule layer** computed from the questions that do work — difficulty flag + tool/harness handling + escalation on failure. Independent benchmark literature (LLMRouterBench, ACL 2026) reaches the same conclusion: most learned routers lose to "best cheap model + escalate on failure." We reproduced that result from scratch, on our own data, before finding theirs.

---

## 3. The scoreboard

| Question | First honest eval | Final state |
|---|---|---|
| `tools_file` | 78.0% | 79.6% — production gate |
| `tools_terminal` | 86.3% | 85.5% — production gate |
| `tools_web` | 87.1% | 85.1% — production gate |
| `needs_verification` | 72.2% | 72.9% — production gate |
| `difficulty` (buckets) | 31% — *below random* | retired, reformulated |
| `difficulty` (0–4 count) | 78.0% (tie vs 77.6%) | **precision flag: 81.2% @ 6.3% coverage** |
| `tier` | 29% | retired → rule layer |

Gate accuracy was verified equivalent between checkpoints in a head-to-head on 255 labeled tasks (323.1 vs 323.6 summed points — a statistical wash) before any production swap.

*Note:* [eval/cold_metrics.json](eval/cold_metrics.json) contains smoke-test metrics on the 40-row synthetic toy corpus to confirm the pipeline runs end-to-end; real evaluation numbers above come from private held-out real task data.

---

## 4. Serving architecture, privacy & outcome ledger

```text
prompt → hook (stdlib, 2s timeout)
          → sidecar (FastAPI, loopback :8777, MLX, ~40ms)
          → judgment dict {tools, verification, difficulty flag p_hard}
          → proxy decision + log (data/proxy_log.jsonl, append-only)
fail-safe: judge unavailable → ROVIK_FALLBACK_ROUTE (default 'local')
```

The fail-safe is the load-bearing wall: the classifier may be wrong, dead, or slow — the assistant must never break.

### 4.0 Privacy and data handling

The isolation boundary is explicit and load-bearing:

- **Local-First:** Requests handled locally never leave the host machine. The classifier sidecar, the local model tier, and the decision/shadow logs all run on loopback, on your hardware.
- **Egress Boundary:** Upstream/frontier endpoints are only contacted when a task is *explicitly escalated* by the routing layer (or when a local tier fails and falls back to escalation). No telemetry, no background requests, nothing leaves the host otherwise.
- **Secret Isolation:** Upstream API keys are held strictly by the egress runner (`serving/proxy.py`, reading `os.environ` first and falling back to `ROVIK_ENV_FILE`). They are never exposed to — and never required by — the local Laya classifier sidecar.
- **Harness Agnostic (`/v1/chat/completions`):** Rovik exposes OpenAI Chat Completions (`POST /v1/chat/completions` + `GET /v1/models`) and is compatible with `/v1/chat/completions` clients/harnesses (including streaming SSE and tool-carrying payloads). The same routing architecture has run daily behind a local-first Hermes agent profile; this repo is its cleaned-up continuation, verified end-to-end from a clean clone but not yet soak-tested on live traffic. Harnesses speaking Anthropic Messages (`POST /v1/messages`, Claude Code) or OpenAI Responses (`POST /v1/responses`, current Codex CLI) are not yet implemented natively (see [§5](#5-status-open-problems--troubleshooting)).

### 4.1 The router (`serving/proxy.py`) — and how to see who answered

The proxy sits on `127.0.0.1:8788`, judges incoming prompts (including tool-carrying harness prompts: simple tool calls go local, hard multi-step work escalates), mints a `decision_id` + `task_text_hash`, and prepends a visual route tag to each reply (`[🟢 LOCAL · 35B-A3B]` vs `[⚡ ESCALATED · GLM 5.3]`, configurable via `ROVIK_TAG_LOCAL` / `ROVIK_TAG_ESC`).

Two hard-won tag rules, learned from the first real session:

1. **Tags are for the human, not the model.** The proxy strips tag lines (`[🟢 ...]`, `[⚡ ...]`, bare `🟢`/`⚡`, and exact configured tags) from assistant conversation history before forwarding upstream. Without this, the model sees tagged history, imitates the tag (double-tagging), and confabulates explanations about a routing layer it cannot see.
2. **Never hardcode model names.** Multi-frontier setups change the escalation target per workload (`ROVIK_ESC_MODEL`); a tag that names a model goes stale the first time you swap. Icons plus a short label you control beat echoing whatever tonight's stack was.

### 4.2 The savings dashboard

Opening the proxy root URL (`http://127.0.0.1:8788/`) serves [serving/dashboard.html](serving/dashboard.html), polling `/stats` and `/health` to display estimated savings (`local_tokens / 1e6 * ROVIK_FRONTIER_PRICE_PER_MTOK`), route mix, actual frontier spend, active fail-safe policy, and recent routing decisions. Estimates count only usage captured from upstream responses — nothing is fabricated.

### 4.3 The outcome ledger — closing the loop asynchronously

Every routing decision mints a `decision_id` (12-hex UUID) alongside `task_text_hash` (SHA-256 prefix) and `routed_to` (`local` / `frontier`), threaded at position 0 of the forwarded system messages so it survives restarts and transcript exports. Offline, two stdlib-only scripts close the loop ([eval/outcome_capture_design.md](eval/outcome_capture_design.md)):

```text
                    ┌──────────────────────────────────────────────────┐
                    │              REQUEST PATH (hot, <40ms judge)     │
                    │                                                  │
  client ──────────►│  proxy.py ── mint decision_id ──┬── local model  │
                    │        │                        └── frontier     │
                    │        ▼                                         │
                    │  data/proxy_log.jsonl (or shadow.py ledger)      │
                    └──────────────┬───────────────────────────────────┘
                                   │  (async — never blocks requests)
                                   ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │                OFFLINE JOIN (cold)                                 │
  │                                                                    │
  │  transcripts/logs ──► outcome_extract.py ──► outcome_events.jsonl  │
  │                        (tool_error, user_retry,                    │
  │                         assistant_error_msg, explicit_success)     │
  │                                      │                             │
  │                                      ▼                             │
  │  outcome_joiner.py: join on decision_id (fallback: task_text_hash  │
  │  ± 5 min window, unambiguous) ──► shadow_log_enriched.jsonl        │
  └────────────────────────────────────────────────────────────────────┘
```

**Why deterministic runtime signals beat LLM-as-a-judge.** A judge model scoring "did the model do well?" re-introduces subjective ambiguity and label drift. A `tool_error` in a transcript or an immediate `user_retry` is a deterministic runtime fact observable without opinions.

The joiner ([serving/outcome_joiner.py](serving/outcome_joiner.py)) is byte-offset-checkpointed, preserves append-only originals, enforces precedence (`tool_error` > `user_retry_after_error` > `assistant_error_msg` > `human_redo` > `user_retry` > `explicit_success`), and quarantines unmatched or ambiguous events to `outcome_unmatched.jsonl`.

---

## 5. Status, open problems & troubleshooting

- **Current status:** Experimental (`v0.2`). Verified on Apple Silicon (`M1`, `64GB`). The full pipeline (prepare corpus → RLCD train + temperature calibration → eval → sweep → serve sidecar + proxy → outcome extract/join) runs locally.
- **Open problem — automated retraining:** A nightly retrain burns ~88 minutes of GPU producing an identical checkpoint when the corpus hasn't shifted. Triggering retraining only when newly enriched rows accumulate, and promoting checkpoints only after out-of-sample evaluation beats production, remains open.
- **Cold starts & MLX memory hygiene:** Initial sidecar startup loads the checkpoint into unified memory; warmup runs automatically during FastAPI lifespan startup. Under varied-length traffic, MLX's Metal allocator pool can fragment (~43 GB cache growth over 100 varied requests measured in [eval/sidecar_memory_audit.md](eval/sidecar_memory_audit.md)); `serving/sidecar.py` clears the Metal cache every `LAYA_CLEAR_CACHE_EVERY` requests (default `10`).
- **Concurrency:** Tuned for single-user local agent loops (requests serialized in the sidecar via `threading.Lock`).
- **Fail-safe behavior:** Judge failures route according to `ROVIK_FALLBACK_ROUTE` (default `local`, or `escalate`). A broken judge costs speed on `local`, never accidental cloud spend.

### Known harness gaps & code backlog

- **Missing `/v1/messages` and `/v1/responses` routes:** [serving/proxy.py](serving/proxy.py) currently exposes `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, `GET /stats`, and `GET /` only. Claude Code (`POST /v1/messages`) and current Codex CLI (`POST /v1/responses`, since `wire_api = "chat"` was removed) require an external protocol adapter until those routes are added.
- **Non-streaming tool-call null content tagging:** In [serving/proxy.py](serving/proxy.py), streaming SSE only prepends `tag` when `delta.content` is non-empty, whereas the non-streaming paths (`local` and `esc`) still inject `f"{tag}\n"` even when `message.content` is `null` or `""` on a pure tool-call response.
- **Dual `system` message prepend on decision ID injection:** `_thread_decision_id()` inserts a second `{"role": "system", ...}` message at index `0`, which strict upstream chat templates that allow only one leading `system` message can reject (`400`); merging into an existing leading system message (or gating injection behind a config flag) is tracked for a future code pass.

---

## 6. The four-model decomposition experiment

Asking four different large models, blind and independently, to "turn subjective difficulty into five objective yes/no questions" produced three convergent axes across all four (`multi_step`, `discovery` vs procedure, open-ended done criteria) plus four unique contributions that survived into the candidate set. Orthogonality mattered: three rewordings of "is it complex?" give false confidence in a vote. The merged set, plus per-question contrastive examples in the rubric ([pipeline/agreement_test.py](pipeline/agreement_test.py), [pipeline/prepare_corpus.py](pipeline/prepare_corpus.py)), is what the teacher labeled against — and the ≥85% agreement gate is what certified it (cutting `done_open` at 79%).

---

## 7. Acknowledging the lineage

Laya is an open-weight (Apache 2.0) small decision model in the family popularized by Jev-style classifiers. JevBench-style leaderboards score these models on generic question suites. This project's core finding is that **leaderboard scores don't transfer to your tasks** — and the fix isn't a better base model, it's your own labels, tested for consistency, evaluated against the dumb strategy, and shipped honestly.
