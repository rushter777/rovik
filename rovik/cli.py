"""Unified CLI entry point for `rovik` (stdlib-only dispatch)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from rovik import __version__
from rovik.config import KNOWN_KEYS, load_config, parse_env_file, resolve_config_path

_HELP_TEXT = """usage: rovik [--config PATH] [--version] <command> [args...]

Rovik — local-first AI model routing switch.

commands:
  init          Interactive setup wizard to generate rovik.env
  config        Show resolved config path and active settings (secrets redacted)
  prepare-data  Inspect, agreement-test, label, and split your task corpus
  proxy         Run the OpenAI-compatible routing proxy (:8788)
  sidecar       Run the Laya MLX judgment sidecar (:8777)
  judge         Judge a single task string via the sidecar
  train         Fine-tune Laya with RLCD on a prepared JSONL split
  eval          Evaluate a fine-tuned checkpoint on an eval JSONL split
  sweep         Sweep difficulty cutoffs for precision vs. coverage
  outcomes      Run offline transcript extraction (`extract`) or ledger join (`join`)
"""

_SECRET_VALUE_SUFFIXES = ("_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD")


def _run_with_argv(prog_name: str, args: list[str], fn) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [prog_name] + list(args)
        rc = fn()
        return int(rc) if isinstance(rc, int) else 0
    finally:
        sys.argv = old_argv


def _is_secret_key_name(key: str) -> bool:
    return any(key.endswith(sfx) for sfx in _SECRET_VALUE_SUFFIXES)


def _cmd_init(args: list[str]) -> int:
    p = argparse.ArgumentParser(prog="rovik init", description="Interactive setup wizard for rovik.env")
    p.add_argument("--output", default="rovik.env", help="Output config path (default: rovik.env)")
    p.add_argument("--non-interactive", action="store_true", help="Write defaults without prompting")
    parsed = p.parse_args(args)
    from rovik.init_wizard import run_wizard

    run_wizard(output_path=parsed.output, non_interactive=parsed.non_interactive)
    return 0


def _cmd_config(resolved_path: str | None) -> int:
    file_vals = parse_env_file(resolved_path) if resolved_path else {}
    print(f"Config file: {resolved_path or '(none — using defaults / process env)'}")
    for key in KNOWN_KEYS:
        if key in os.environ and (key not in file_vals or os.environ[key] != file_vals[key]):
            src = "env"
            raw_val = os.environ[key]
        elif key in file_vals:
            src = "file"
            raw_val = file_vals[key]
        else:
            src = "default"
            raw_val = ""

        if _is_secret_key_name(key) and raw_val:
            shown = "*** (redacted)"
        elif key in ("ROVIK_ESC_KEYNAME", "ROVIK_TEACHER_KEYNAME") and raw_val:
            status = "set" if os.environ.get(raw_val, "").strip() else "unset"
            shown = f"{raw_val} (env: {status})"
        else:
            shown = raw_val if raw_val != "" else "(unset)"
        print(f"  {key:<30s} [{src:<7s}] = {shown}")
    return 0


def _cmd_judge(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print("Usage: rovik judge '<task description>'")
        return 0 if args else 1
    from serving import hook

    task = " ".join(args)
    cfg = hook.apply(hook.judge(task))
    print(json.dumps(cfg, indent=2))
    return 0


def _cmd_outcomes(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print("Usage: rovik outcomes {extract|join} [args...]")
        return 0 if args else 1
    sub, rest = args[0], args[1:]
    if sub == "extract":
        from serving import outcome_extract

        return _run_with_argv("rovik outcomes extract", rest, outcome_extract.main)
    if sub == "join":
        from serving import outcome_joiner

        return _run_with_argv("rovik outcomes join", rest, outcome_joiner.main)
    print(f"Unknown outcomes subcommand: {sub}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    config_arg: str | None = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--config":
            if i + 1 >= len(args):
                print("Error: --config requires a path", file=sys.stderr)
                return 2
            config_arg = args[i + 1]
            del args[i : i + 2]
            continue
        if tok.startswith("--config="):
            config_arg = tok.split("=", 1)[1]
            del args[i]
            continue
        if tok == "--version":
            print(f"rovik {__version__}")
            return 0
        if tok in ("-h", "--help") and i == 0:
            print(_HELP_TEXT.rstrip())
            return 0
        break

    if not args:
        print(_HELP_TEXT.rstrip())
        return 0

    cmd, rest = args[0], args[1:]
    if cmd == "init":
        return _cmd_init(rest)

    if config_arg is not None and resolve_config_path(config_arg) is None:
        print(f"Error: config file not found: {config_arg}", file=sys.stderr)
        return 2

    resolved_path, _ = load_config(config_arg)

    if cmd == "config":
        return _cmd_config(resolved_path)
    if cmd == "prepare-data":
        from pipeline import prepare_corpus

        return prepare_corpus.main(rest)
    if cmd == "proxy":
        import uvicorn
        from serving import proxy

        uvicorn.run(proxy.app, host=proxy.PROXY_HOST, port=proxy.PROXY_PORT)
        return 0
    if cmd == "sidecar":
        from serving import sidecar

        sidecar.main()
        return 0
    if cmd == "judge":
        return _cmd_judge(rest)
    if cmd == "train":
        from pipeline import train_rlcd

        return _run_with_argv("rovik train", rest, train_rlcd.main)
    if cmd == "eval":
        from pipeline import eval_laya

        return _run_with_argv("rovik eval", rest, eval_laya.main)
    if cmd == "sweep":
        from pipeline import sweep

        return _run_with_argv("rovik sweep", rest, sweep.main)
    if cmd == "outcomes":
        return _cmd_outcomes(rest)

    print(f"Unknown command: {cmd}\n\n{_HELP_TEXT.rstrip()}", file=sys.stderr)
    return 2
