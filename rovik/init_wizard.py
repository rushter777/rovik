"""Interactive setup wizard (`rovik init`) writing `rovik.env` (stdlib-only)."""

from __future__ import annotations

import os

from rovik.config import parse_env_file, write_env_file

LOCAL_PRESETS = {
    "1": ("llama.cpp (llama-server)", "http://127.0.0.1:8082/v1/chat/completions", "local-35b"),
    "2": ("Ollama", "http://127.0.0.1:11434/v1/chat/completions", "qwen2.5:14b"),
    "3": ("LM Studio", "http://127.0.0.1:1234/v1/chat/completions", "local-model"),
    "4": ("vLLM / MLX-LM", "http://127.0.0.1:8000/v1/chat/completions", "local-model"),
    "5": ("Custom OpenAI-compatible URL", "http://127.0.0.1:8081/v1/chat/completions", "local-model"),
}

FRONTIER_PRESETS = {
    "1": ("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", "z-ai/glm-5.3", "OPENROUTER_API_KEY", "3.0"),
    "2": ("OpenAI", "https://api.openai.com/v1/chat/completions", "gpt-5", "OPENAI_API_KEY", "5.0"),
    "3": ("Custom OpenAI-compatible endpoint", "https://openrouter.ai/api/v1/chat/completions", "frontier-model", "OPENAI_API_KEY", "3.0"),
}


def _match_preset(presets: dict[str, tuple], current_url: str, fallback_key: str) -> str:
    if not current_url:
        return "1"
    for key, entry in presets.items():
        if entry[1] == current_url:
            return key
    return fallback_key


def run_wizard(
    output_path: str = "rovik.env",
    *,
    non_interactive: bool = False,
    input_fn=input,
    print_fn=print,
) -> dict[str, str]:
    """Prompt for local tier, frontier tier, teacher, and corpus/sidecar paths, writing rovik.env."""
    expanded_out = os.path.expanduser(output_path)
    existing = parse_env_file(expanded_out) if os.path.isfile(expanded_out) else {}

    def _ask(prompt_text: str, default: str) -> str:
        if non_interactive:
            return default
        suffix = f" [{default}]" if default != "" else " []"
        ans = input_fn(f"{prompt_text}{suffix}: ").strip()
        return default if ans == "" else ans

    print_fn("Rovik Setup — configure local tier, frontier escalation, teacher, and corpus paths.")
    print_fn(f"Writing config to: {expanded_out}\n")

    # 1. Local Tier
    print_fn("[1/4] Local Model (Tier 1 — fast/free local inference)")
    for k, (label, url, _) in LOCAL_PRESETS.items():
        print_fn(f"  {k}) {label} ({url})")
    cur_local_url = existing.get("ROVIK_LOCAL_URL", "")
    default_local_choice = _match_preset(LOCAL_PRESETS, cur_local_url, "5")
    local_choice = _ask("Choose local backend", default_local_choice)
    if local_choice not in LOCAL_PRESETS:
        local_choice = default_local_choice
    _, preset_local_url, preset_local_model = LOCAL_PRESETS[local_choice]

    use_existing_local = bool(cur_local_url) and local_choice == default_local_choice
    local_url = _ask(
        "Local OpenAI-compatible chat completions URL",
        cur_local_url if use_existing_local else preset_local_url,
    )
    local_model = _ask(
        "Local model name",
        existing.get("ROVIK_LOCAL_MODEL", preset_local_model) if use_existing_local else preset_local_model,
    )
    tag_local = _ask(
        "Reply tag for local answers",
        existing.get("ROVIK_TAG_LOCAL", "[🟢 LOCAL · 35B-A3B]"),
    )

    # 2. Frontier Tier
    print_fn("\n[2/4] Frontier Model (Tier 2 — selective escalation)")
    for k, (label, url, _, _, _) in FRONTIER_PRESETS.items():
        print_fn(f"  {k}) {label} ({url})")
    cur_esc_url = existing.get("ROVIK_ESC_URL", "")
    default_esc_choice = _match_preset(FRONTIER_PRESETS, cur_esc_url, "3")
    esc_choice = _ask("Choose frontier provider", default_esc_choice)
    if esc_choice not in FRONTIER_PRESETS:
        esc_choice = default_esc_choice
    _, preset_esc_url, preset_esc_model, preset_keyname, preset_price = FRONTIER_PRESETS[esc_choice]

    use_existing_esc = bool(cur_esc_url) and esc_choice == default_esc_choice
    esc_url = _ask(
        "Frontier chat completions URL",
        cur_esc_url if use_existing_esc else preset_esc_url,
    )
    esc_model = _ask(
        "Frontier model ID",
        existing.get("ROVIK_ESC_MODEL", preset_esc_model) if use_existing_esc else preset_esc_model,
    )
    esc_keyname = _ask(
        "Env var name holding your frontier API key (secret itself is NOT stored here)",
        existing.get("ROVIK_ESC_KEYNAME", preset_keyname) if use_existing_esc else preset_keyname,
    )
    env_file = _ask(
        "Optional fallback .env file for API keys",
        existing.get("ROVIK_ENV_FILE", "~/.hermes/.env"),
    )
    frontier_price = _ask(
        "Estimated frontier price ($ per 1M tokens, for savings dashboard)",
        existing.get("ROVIK_FRONTIER_PRICE_PER_MTOK", preset_price) if use_existing_esc else preset_price,
    )
    tag_esc = _ask(
        "Reply tag for escalated answers",
        existing.get("ROVIK_TAG_ESC", "[⚡ ESCALATED · GLM 5.3]"),
    )

    # 3. Labeling Teacher
    print_fn("\n[3/4] Labeling Teacher (training-time only; >=30B local or frontier API)")
    print_fn("  1) Same as local model")
    print_fn("  2) Same as frontier model")
    print_fn("  3) Custom endpoint")
    cur_teacher_url = existing.get("ROVIK_TEACHER_URL", "")
    if cur_teacher_url and cur_teacher_url == esc_url:
        default_teacher_choice = "2"
    elif cur_teacher_url and cur_teacher_url != local_url:
        default_teacher_choice = "3"
    else:
        default_teacher_choice = "1"
    teacher_choice = _ask("Choose labeling teacher", default_teacher_choice)
    if teacher_choice not in ("1", "2", "3"):
        teacher_choice = default_teacher_choice

    if teacher_choice == "1":
        teacher_url = local_url
        teacher_model = local_model
        teacher_keyname = ""
    elif teacher_choice == "2":
        teacher_url = esc_url
        teacher_model = esc_model
        teacher_keyname = esc_keyname
    else:
        teacher_url = _ask("Teacher chat completions URL", cur_teacher_url or local_url)
        teacher_model = _ask("Teacher model ID", existing.get("ROVIK_TEACHER_MODEL", local_model))
        teacher_keyname = _ask(
            "Teacher API key env var name (blank if local)",
            existing.get("ROVIK_TEACHER_KEYNAME", ""),
        )

    # 4. Data, Checkpoint & Safety
    print_fn("\n[4/4] Corpus, Checkpoint & Routing Settings")
    corpus_path = _ask(
        "Default path to your raw prompt corpus (.txt/.csv/.jsonl, optional)",
        existing.get("ROVIK_CORPUS_PATH", ""),
    )
    model_path = _ask(
        "Laya decision checkpoint directory",
        existing.get("LAYA_MODEL_PATH", "checkpoints/cold-test"),
    )
    thresh_hard = _ask(
        "Difficulty escalation cutoff P(hard) (lower = escalate more, e.g. 0.5 for 9B/14B)",
        existing.get("LAYA_THRESH_HARD", "0.8"),
    )
    fallback_route = _ask(
        "Fail-safe route if sidecar is down (local or escalate)",
        existing.get("ROVIK_FALLBACK_ROUTE", "local"),
    ).strip().lower()
    if fallback_route not in ("local", "escalate"):
        fallback_route = "local"
    log_dir = _ask("Data & log directory", existing.get("ROVIK_LOG_DIR", "data"))
    proxy_host = _ask("Proxy bind host", existing.get("ROVIK_PROXY_HOST", "127.0.0.1"))
    proxy_port = _ask("Proxy bind port", existing.get("ROVIK_PROXY_PORT", "8788"))
    sidecar_host = existing.get("LAYA_SIDECAR_HOST", "127.0.0.1")
    sidecar_port = _ask("Sidecar bind port", existing.get("LAYA_SIDECAR_PORT", "8777"))
    sidecar_url = f"http://{sidecar_host}:{sidecar_port}/predict"

    values = {
        "ROVIK_LOCAL_URL": local_url,
        "ROVIK_LOCAL_MODEL": local_model,
        "ROVIK_TAG_LOCAL": tag_local,
        "ROVIK_ESC_URL": esc_url,
        "ROVIK_ESC_MODEL": esc_model,
        "ROVIK_ESC_KEYNAME": esc_keyname,
        "ROVIK_ENV_FILE": env_file,
        "ROVIK_FRONTIER_PRICE_PER_MTOK": frontier_price,
        "ROVIK_TAG_ESC": tag_esc,
        "ROVIK_TEACHER_URL": teacher_url,
        "ROVIK_TEACHER_MODEL": teacher_model,
        "ROVIK_TEACHER_KEYNAME": teacher_keyname,
        "ROVIK_CORPUS_PATH": corpus_path,
        "LAYA_MODEL_PATH": model_path,
        "LAYA_SIDECAR_HOST": sidecar_host,
        "LAYA_SIDECAR_PORT": sidecar_port,
        "LAYA_SIDECAR_URL": sidecar_url,
        "LAYA_CLEAR_CACHE_EVERY": existing.get("LAYA_CLEAR_CACHE_EVERY", "10"),
        "LAYA_THRESH_HARD": thresh_hard,
        "LAYA_THRESH_TOOLS": existing.get("LAYA_THRESH_TOOLS", "0.5"),
        "LAYA_THRESH_VERIFY": existing.get("LAYA_THRESH_VERIFY", "0.5"),
        "ROVIK_PROXY_HOST": proxy_host,
        "ROVIK_PROXY_PORT": proxy_port,
        "ROVIK_JUDGE_URL": sidecar_url,
        "ROVIK_FALLBACK_ROUTE": fallback_route,
        "ROVIK_LOG_DIR": log_dir,
    }

    written = write_env_file(
        expanded_out,
        values,
        header_comments=[
            "Generated by `rovik init`. Keep this file private (gitignored).",
            "API key values are read from your environment or ROVIK_ENV_FILE, not stored here.",
        ],
    )

    print_fn(f"\nSaved Rovik configuration to {written} (mode 0600).")
    print_fn("Next steps:")
    if corpus_path:
        print_fn(f"  1. Inspect your corpus: rovik prepare-data --input {corpus_path} --inspect")
    else:
        print_fn("  1. Inspect a corpus:    rovik prepare-data --input <your_prompts.jsonl> --inspect")
    print_fn("  2. Start the sidecar:   rovik sidecar")
    print_fn("  3. Start the proxy:     rovik proxy")
    return values
