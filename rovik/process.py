"""Background process manager (`rovik start`, `rovik stop`, `rovik status`) (stdlib-only)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from rovik.config import REPO_ROOT, detect_env_overrides, diagnose_key_source
from rovik.preflight import format_preflight, run_preflight


def get_log_dir() -> str:
    """Return expanded log/pid directory (ROVIK_LOG_DIR or <repo>/data)."""
    raw = os.environ.get("ROVIK_LOG_DIR", "").strip()
    return os.path.expanduser(raw) if raw else os.path.join(REPO_ROOT, "data")


def pid_file_path(service: str, log_dir: str | None = None) -> str:
    """Return pidfile path for *service* ('sidecar' or 'proxy')."""
    base = log_dir if log_dir is not None else get_log_dir()
    return os.path.join(base, f"{service}.pid")


def log_file_path(service: str, log_dir: str | None = None) -> str:
    """Return log file path for *service* ('sidecar' or 'proxy')."""
    base = log_dir if log_dir is not None else get_log_dir()
    return os.path.join(base, f"{service}.log")


def get_endpoints() -> dict[str, str | int]:
    """Resolve sidecar and proxy bind endpoints from current environment."""
    sidecar_host = os.environ.get("LAYA_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    sidecar_port = int(os.environ.get("LAYA_SIDECAR_PORT", "8777").strip() or "8777")
    proxy_host = os.environ.get("ROVIK_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    proxy_port = int(os.environ.get("ROVIK_PROXY_PORT", "8788").strip() or "8788")
    return {
        "sidecar_host": sidecar_host,
        "sidecar_port": sidecar_port,
        "sidecar_health_url": f"http://{sidecar_host}:{sidecar_port}/health",
        "proxy_host": proxy_host,
        "proxy_port": proxy_port,
        "proxy_health_url": f"http://{proxy_host}:{proxy_port}/health",
        "proxy_stats_url": f"http://{proxy_host}:{proxy_port}/stats",
        "proxy_base_url": f"http://{proxy_host}:{proxy_port}/v1",
        "proxy_dashboard_url": f"http://{proxy_host}:{proxy_port}/",
    }


def _read_pid(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(path: str, pid: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{pid}\n")


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _find_pids_by_port(port: int) -> list[int]:
    """Return listening TCP PIDs bound to *port* via `lsof` (best-effort)."""
    try:
        proc = subprocess.run(
            ["lsof", "-ti", f"tcp:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    pids: list[int] = []
    for tok in (proc.stdout or "").split():
        try:
            pid = int(tok.strip())
            if pid > 0 and pid not in pids:
                pids.append(pid)
        except ValueError:
            continue
    return pids


def _is_rovik_process(pid: int) -> bool:
    """Return True if *pid* command line looks like a Rovik sidecar or proxy worker."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    cmd = (proc.stdout or "").lower()
    markers = ("rovik", "serving.proxy", "serving.sidecar", "serving/proxy.py", "serving/sidecar.py")
    return any(m in cmd for m in markers)


def _terminate_pid(pid: int, timeout_s: float = 5.0) -> bool:
    """Send SIGTERM (then SIGKILL after *timeout_s*) to *pid*. Returns True if exited."""
    if not _pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False

    deadline = time.monotonic() + max(0.2, timeout_s)
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return not _pid_alive(pid)
    time.sleep(0.1)
    return not _pid_alive(pid)


def _probe_json(url: str, timeout_s: float = 2.0) -> tuple[int | None, dict | None, str | None]:
    """GET JSON endpoint, returning (http_status, parsed_json_or_None, error_or_None)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw), None
            except ValueError as exc:
                return resp.status, None, f"invalid JSON from {url}: {exc}"
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        finally:
            try:
                exc.close()
            except Exception:
                pass
        parsed = None
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
        return exc.code, parsed, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return None, None, str(exc)


def _tail_log(path: str, lines: int = 15) -> str:
    if not os.path.isfile(path):
        return "(no log output yet)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            buf = f.readlines()
        tail = "".join(buf[-lines:]).strip()
        return tail or "(log file empty)"
    except OSError as exc:
        return f"(unable to read log: {exc})"


def _spawn_service(
    service: str,
    *,
    config_path: str | None,
    child_env: dict[str, str],
    log_dir: str,
) -> subprocess.Popen:
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_file_path(service, log_dir)
    pid_path = pid_file_path(service, log_dir)
    cmd = [sys.executable, "-m", "rovik"]
    if config_path:
        cmd.extend(["--config", config_path])
    cmd.append(service)

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fh = open(log_path, "a", encoding="utf-8")
    try:
        fh.write(f"\n--- rovik {service} starting at {stamp} ---\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        fh.close()

    _write_pid(pid_path, proc.pid)
    return proc


def _wait_for_service(
    service: str,
    proc: subprocess.Popen,
    health_url: str,
    log_path: str,
    *,
    timeout_s: float,
    fail_on_503_error: bool = False,
) -> tuple[dict | None, str | None]:
    """Wait for spawned *service* to report healthy HTTP 200 JSON, or return failure text."""
    deadline = time.monotonic() + max(0.5, timeout_s)
    last_err = "timed out waiting for health endpoint"
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return None, f"{service} exited immediately (code {rc}). Recent log ({log_path}):\n{_tail_log(log_path)}"
        status, payload, err = _probe_json(health_url, timeout_s=1.0)
        if status == 200 and isinstance(payload, dict) and payload.get("ok"):
            return payload, None
        if fail_on_503_error and status == 503 and isinstance(payload, dict) and payload.get("error"):
            return None, f"{service} failed model load: {payload['error']}\nLog ({log_path}):\n{_tail_log(log_path)}"
        if err:
            last_err = err
        time.sleep(0.25)
    return None, f"{service} did not become healthy within {timeout_s:.1f}s ({last_err}). Log ({log_path}):\n{_tail_log(log_path)}"


def _startup_env_warnings() -> list[str]:
    warnings_list: list[str] = []
    key_name = os.environ.get("ROVIK_ESC_KEYNAME", "OPENROUTER_API_KEY").strip()
    if key_name:
        kdiag = diagnose_key_source(key_name)
        if kdiag["warning"]:
            warnings_list.append(kdiag["warning"])
    for item in detect_env_overrides():
        k = item["key"]
        if k.startswith("ROVIK_") or k.startswith("LAYA_"):
            warnings_list.append(item["warning"])
    return warnings_list


def start_services(
    *,
    config_path: str | None = None,
    base_env: dict[str, str] | None = None,
    wait: bool = True,
    timeout_s: float = 30.0,
    skip_preflight: bool = False,
    print_fn=print,
) -> int:
    """Preflight and launch backgrounded sidecar and proxy processes."""
    if not skip_preflight:
        res = run_preflight(check_serving=True)
        if not res.ok:
            print_fn("Rovik preflight failed — refusing to start:")
            print_fn(format_preflight(res))
            return 1

    for warn in _startup_env_warnings():
        print_fn(f"WARNING: {warn}")

    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    ep = get_endpoints()
    child_env = dict(base_env) if base_env is not None else os.environ.copy()
    if skip_preflight:
        child_env["ROVIK_SKIP_PREFLIGHT"] = "1"

    # 1. Start Laya sidecar
    sidecar_pid_file = pid_file_path("sidecar", log_dir)
    sidecar_log = log_file_path("sidecar", log_dir)
    sidecar_health: dict | None = None
    sidecar_pid: int | None = None

    s_status, s_payload, _ = _probe_json(str(ep["sidecar_health_url"]), timeout_s=1.0)
    if s_status == 200 and isinstance(s_payload, dict) and s_payload.get("ok"):
        sidecar_health = s_payload
        existing_pid = _read_pid(sidecar_pid_file)
        if _pid_alive(existing_pid):
            sidecar_pid = existing_pid
        else:
            port_pids = _find_pids_by_port(int(ep["sidecar_port"]))
            sidecar_pid = port_pids[0] if port_pids else None
        print_fn(f"Sidecar already running at {ep['sidecar_health_url']}")
    else:
        existing_pid = _read_pid(sidecar_pid_file)
        if not _pid_alive(existing_pid):
            _remove_file(sidecar_pid_file)
        port_pids = _find_pids_by_port(int(ep["sidecar_port"]))
        if port_pids:
            print_fn(
                f"Error: port {ep['sidecar_port']} is already in use by PID(s) {port_pids} "
                f"and {ep['sidecar_health_url']} is not healthy. Run `rovik stop --force-port` or free the port."
            )
            return 1
        s_proc = _spawn_service("sidecar", config_path=config_path, child_env=child_env, log_dir=log_dir)
        sidecar_pid = s_proc.pid
        if wait:
            sidecar_health, err = _wait_for_service(
                "sidecar",
                s_proc,
                str(ep["sidecar_health_url"]),
                sidecar_log,
                timeout_s=timeout_s,
                fail_on_503_error=True,
            )
            if err:
                _terminate_pid(s_proc.pid, timeout_s=2.0)
                _remove_file(sidecar_pid_file)
                print_fn(f"Error starting {err}")
                return 1

    # 2. Start Rovik proxy
    proxy_pid_file = pid_file_path("proxy", log_dir)
    proxy_log = log_file_path("proxy", log_dir)
    proxy_health: dict | None = None
    proxy_pid: int | None = None

    p_status, p_payload, _ = _probe_json(str(ep["proxy_health_url"]), timeout_s=1.0)
    if p_status == 200 and isinstance(p_payload, dict) and p_payload.get("ok"):
        proxy_health = p_payload
        existing_pid = _read_pid(proxy_pid_file)
        if _pid_alive(existing_pid):
            proxy_pid = existing_pid
        else:
            port_pids = _find_pids_by_port(int(ep["proxy_port"]))
            proxy_pid = port_pids[0] if port_pids else None
        print_fn(f"Proxy already running at {ep['proxy_base_url']}")
    else:
        existing_pid = _read_pid(proxy_pid_file)
        if not _pid_alive(existing_pid):
            _remove_file(proxy_pid_file)
        port_pids = _find_pids_by_port(int(ep["proxy_port"]))
        if port_pids:
            print_fn(
                f"Error: port {ep['proxy_port']} is already in use by PID(s) {port_pids} "
                f"and {ep['proxy_health_url']} is not healthy. Run `rovik stop --force-port` or free the port."
            )
            return 1
        p_proc = _spawn_service("proxy", config_path=config_path, child_env=child_env, log_dir=log_dir)
        proxy_pid = p_proc.pid
        if wait:
            proxy_health, err = _wait_for_service(
                "proxy",
                p_proc,
                str(ep["proxy_health_url"]),
                proxy_log,
                timeout_s=min(timeout_s, 15.0),
                fail_on_503_error=False,
            )
            if err:
                _terminate_pid(p_proc.pid, timeout_s=2.0)
                _remove_file(proxy_pid_file)
                print_fn(f"Error starting {err}\nNote: sidecar may still be running — run `rovik stop` to stop it.")
                return 1

    print_fn("\nRovik started:")
    print_fn(f"  Sidecar:   {ep['sidecar_health_url']} (pid={sidecar_pid or '?'}, log={sidecar_log})")
    print_fn(f"  Proxy API: {ep['proxy_base_url']} (pid={proxy_pid or '?'}, log={proxy_log})")
    print_fn(f"  Dashboard: {ep['proxy_dashboard_url']}")

    if isinstance(proxy_health, dict) and isinstance(proxy_health.get("esc_key"), dict):
        ek = proxy_health["esc_key"]
        src_txt = ek.get("source", "unknown")
        if src_txt == "env-file" and ek.get("env_file"):
            src_txt = f"env-file ({ek['env_file']})"
        print_fn(f"  Frontier key ({ek.get('name', '?')}): source={src_txt}")

    all_warnings: list[str] = []
    for h in (sidecar_health, proxy_health):
        if isinstance(h, dict):
            for w in h.get("warnings") or []:
                if w not in all_warnings:
                    all_warnings.append(w)
    for w in all_warnings:
        print_fn(f"  WARNING: {w}")

    print_fn("\nVerify anytime with: rovik status  |  Stop both with: rovik stop")
    return 0


def stop_services(
    *,
    force_port: bool = False,
    timeout_s: float = 5.0,
    print_fn=print,
) -> int:
    """Stop backgrounded proxy and sidecar by pidfile, with port fallback."""
    log_dir = get_log_dir()
    ep = get_endpoints()

    targets = (
        ("proxy", int(ep["proxy_port"])),
        ("sidecar", int(ep["sidecar_port"])),
    )
    for service, port in targets:
        pid_path = pid_file_path(service, log_dir)
        pid = _read_pid(pid_path)
        stopped_msgs: list[str] = []

        if pid is not None and _pid_alive(pid):
            ok = _terminate_pid(pid, timeout_s=timeout_s)
            stopped_msgs.append(f"stopped (pid {pid})" if ok else f"failed to stop (pid {pid})")
        _remove_file(pid_path)

        for listener in _find_pids_by_port(port):
            if listener == pid:
                continue
            if force_port or _is_rovik_process(listener):
                ok = _terminate_pid(listener, timeout_s=timeout_s)
                stopped_msgs.append(
                    f"stopped by port {port} (pid {listener})" if ok else f"failed to stop port {port} (pid {listener})"
                )

        if stopped_msgs:
            print_fn(f"  {service}: {', '.join(stopped_msgs)}")
        else:
            print_fn(f"  {service}: not running")
    return 0


def collect_status() -> dict:
    """Probe sidecar, proxy, pidfiles, stats, and startup/runtime warnings."""
    log_dir = get_log_dir()
    ep = get_endpoints()

    def _service_info(name: str, port: int, health_url: str) -> dict:
        pid_path = pid_file_path(name, log_dir)
        log_path = log_file_path(name, log_dir)
        pid = _read_pid(pid_path)
        alive = _pid_alive(pid)
        if not alive:
            port_pids = _find_pids_by_port(port)
            pid = port_pids[0] if port_pids else None
            alive = pid is not None
        http_status, health_json, err = _probe_json(health_url, timeout_s=2.0)
        ok = http_status == 200 and isinstance(health_json, dict) and bool(health_json.get("ok"))
        return {
            "ok": ok,
            "pid": pid if alive else None,
            "pid_alive": alive,
            "pid_file": pid_path,
            "log_file": log_path,
            "health_url": health_url,
            "http_status": http_status,
            "health": health_json,
            "error": err if not ok else None,
        }

    sidecar_info = _service_info("sidecar", int(ep["sidecar_port"]), str(ep["sidecar_health_url"]))
    proxy_info = _service_info("proxy", int(ep["proxy_port"]), str(ep["proxy_health_url"]))

    stats_data = None
    if proxy_info["ok"]:
        st_code, st_json, _ = _probe_json(str(ep["proxy_stats_url"]), timeout_s=2.0)
        if st_code == 200 and isinstance(st_json, dict):
            stats_data = st_json

    combined_warnings: list[str] = []
    for svc_key in (sidecar_info, proxy_info):
        h = svc_key.get("health")
        if isinstance(h, dict):
            for w in h.get("warnings") or []:
                if w not in combined_warnings:
                    combined_warnings.append(w)

    if not proxy_info["ok"]:
        for w in _startup_env_warnings():
            if w not in combined_warnings:
                combined_warnings.append(w)

    return {
        "ok": bool(sidecar_info["ok"] and proxy_info["ok"]),
        "sidecar": sidecar_info,
        "proxy": proxy_info,
        "stats": stats_data,
        "warnings": combined_warnings,
    }


def status_services(*, as_json: bool = False, print_fn=print) -> int:
    """Print Rovik sidecar & proxy health plus captured runtime warnings."""
    info = collect_status()
    if as_json:
        print_fn(json.dumps(info, indent=2))
        return 0 if info["ok"] else 1

    sc = info["sidecar"]
    sc_state = "healthy" if sc["ok"] else ("unloaded (503)" if sc["http_status"] == 503 else "down")
    print_fn(f"Sidecar: {sc_state} ({sc['health_url']})")
    if sc["pid"]:
        print_fn(f"  PID: {sc['pid']}")
    print_fn(f"  Log: {sc['log_file']}")
    if isinstance(sc.get("health"), dict):
        sh = sc["health"]
        if sh.get("ok"):
            print_fn(
                f"  Model: {sh.get('model')} (backend={sh.get('backend') or 'laya_mlx'}, "
                f"requests={sh.get('requests', 0)})"
            )
        elif sh.get("error"):
            print_fn(f"  Error: {sh['error']}")
    elif sc.get("error"):
        print_fn(f"  Error: {sc['error']}")

    px = info["proxy"]
    px_state = "healthy" if px["ok"] else "down"
    print_fn(f"\nProxy:   {px_state} ({px['health_url']})")
    if px["pid"]:
        print_fn(f"  PID: {px['pid']}")
    print_fn(f"  Log: {px['log_file']}")
    if isinstance(px.get("health"), dict):
        ph = px["health"]
        print_fn(
            f"  Local tier: {ph.get('local')} | Judge: {ph.get('judge')} | "
            f"Fail-safe: {ph.get('fallback_route')} | Escalation errors: {ph.get('esc_failure_policy', 'fail-loud')}"
        )
        ek = ph.get("esc_key")
        if isinstance(ek, dict):
            src = ek.get("source", "unknown")
            if src == "env-file" and ek.get("env_file"):
                src = f"env-file ({ek['env_file']})"
            print_fn(f"  Frontier key ({ek.get('name', '?')}): source={src}")
    elif px.get("error"):
        print_fn(f"  Error: {px['error']}")

    if isinstance(info.get("stats"), dict):
        st = info["stats"]
        print_fn(
            f"\nTraffic: {st.get('decisions_total', 0)} decisions "
            f"(local={st.get('local_count', 0)}, escalated={st.get('escalated_count', 0)}, "
            f"est_savings=${st.get('savings_usd_est', 0):.4f})"
        )

    if info["warnings"]:
        print_fn("\nWarnings:")
        for w in info["warnings"]:
            print_fn(f"  [!] {w}")

    return 0 if info["ok"] else 1
