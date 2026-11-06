"""brain.py: lifecycle of the BRAIN SERVER (vllm-mlx / Qwen).

Used by (1) the vision handoff (vision.read_image_auto): when 24 GB cannot hold the brain
and the image reader at the same time, the brain is PAUSED for a moment, the image is read, and the
brain is REACTIVATED. The 4 atomic operations live here; the orchestration (with the GOLDEN RULE: the
brain ALWAYS comes back) lives in vision.read_image_auto. And (2) the web STATUS PANEL:
`served_model` is its health-check with the observed truth (which model /v1/models serves).

The handoff does not run if there is enough RAM or if it is disabled by config.

It is also the launcher (`main`, the `mlx-agent-brain` command): it starts `vllm-mlx serve` in the
foreground with the model and flags from config.env, on loopback only.
"""

import json
import os
import shutil
import socket
import subprocess  # nosec B404: relaunches this module (fixed arg list, no shell)
import sys
import time
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:  # psutil is a project dependency; without it, stop() cannot locate the process
    psutil = None

ROOT = Path(__file__).resolve().parents[2]  # the repo root (config.env lives there)
_PATTERN = "vllm-mlx"  # marker of the brain server process in its cmdline
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"


def _host_port(cfg):
    return (cfg.get("SERVER_HOST") or "127.0.0.1"), (str(cfg.get("SERVER_PORT") or "8000"))


def is_alive(cfg, timeout=4):
    """Does the brain server respond? (health-check via /v1/models). True/False, never raises."""
    host, port = _host_port(cfg)
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=timeout) as r:  # nosec B310: fixed loopback
            return r.status == 200
    except Exception:  # noqa: BLE001 - any failure means not alive
        return False


def served_model(cfg, timeout=2):
    """Which model(s) does the brain serve RIGHT NOW? (GET /v1/models, the same URL as the
    health-check). None = the brain does NOT respond (down). OBSERVED truth, not config: if the server
    serves a different model than config.env says, this returns the real one. Never raises."""
    host, port = _host_port(cfg)
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=timeout) as r:  # nosec B310: fixed loopback
            if r.status != 200:
                return None
            data = json.loads(r.read(65536).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - any failure means the brain is down/unresponsive
        return None
    names = [str(m.get("id") or "") for m in (data.get("data") or []) if isinstance(m, dict)]
    return ", ".join(n for n in names if n)  # "" = alive but with no models listed (rare; honest)


def _is_the_server(cmdline):
    """Is THIS process the brain server? One of its first arguments IS the vllm-mlx executable and `serve` follows
    (e.g. `python .../bin/vllm-mlx serve model ...`). Before, the command line only had to CONTAIN both words, so
    stop() also closed a terminal or a script that mentioned them."""
    args = list(cmdline or [])
    return any(Path(a).name == _PATTERN and args[i + 1 : i + 2] == ["serve"] for i, a in enumerate(args[:3]))


def _brain_pids():
    """PIDs of the brain server processes (vllm-mlx serve). [] if none or no psutil."""
    if psutil is None:
        return []
    pids = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if _is_the_server(p.info.get("cmdline")):
                pids.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):  # noqa: BLE001
            continue
    return pids


def stop(cfg, timeout=20):
    """Stops the brain server and WAITS for it to die (free port + settle to reclaim RAM).
    Returns True if the brain was effectively stopped; False if it could not be (the caller must NOT
    start a 2nd brain). Never raises."""
    if psutil is None:
        return False
    pids = _brain_pids()
    if not pids and not is_alive(cfg, timeout=2):
        return True  # already stopped
    for pid in pids:
        try:
            psutil.Process(pid).terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:  # noqa: BLE001
            continue
    # wait for a clean shutdown; if it resists, SIGKILL
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not is_alive(cfg, timeout=2) and not _brain_pids():
            break
        time.sleep(0.5)
    for pid in _brain_pids():  # the stubborn ones get a hard kill
        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):  # noqa: BLE001
            continue
    time.sleep(2)  # settle: give the OS a moment to reclaim the model's RAM
    return not is_alive(cfg, timeout=2)


def _vllm_mlx():
    """The vllm-mlx executable: the one installed next to this Python (the project's .venv), else the one
    on PATH. None if it is not installed."""
    here = Path(sys.executable).with_name("vllm-mlx")
    if here.is_file() and os.access(here, os.X_OK):
        return str(here)
    return shutil.which("vllm-mlx")


def can_start(cfg=None):
    """Level 20 (R2): is there a way to turn the brain back on? (the model server is installed). The vision
    handoff does NOT stop a brain it would then be unable to start again."""
    return _vllm_mlx() is not None


def start(cfg):
    """Starts the brain server (this module's `main`, which reads config.env for the correct model/flags) in
    a DETACHED process (start_new_session: survives the web process). Returns the Popen (or None on failure)."""
    if not can_start(cfg):
        return None
    try:
        return subprocess.Popen(  # noqa: S603 - fixed argument list (this Python, this module), no shell  # nosec B603
            [sys.executable, "-m", __name__],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001 - if it does not start, the health-check detects it and reports honestly
        return None


def wait_for_health(cfg, timeout=150):
    """Waits (health-check) for the brain to respond again after starting. True if it came back in time."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_alive(cfg, timeout=3):
            return True
        time.sleep(2)
    return False


def server_args(cfg):
    """Arguments of `vllm-mlx serve`, all from config.env (nothing hardcoded)."""
    host, port = _host_port(cfg)
    args = [
        "serve",
        cfg.get("MLX_MODEL") or _DEFAULT_MODEL,
        "--served-model-name",
        cfg.get("SERVED_MODEL_NAME") or "qwen-coder",
        "--host",
        host,
        "--port",
        port,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        cfg.get("TOOL_CALL_PARSER") or "qwen",
        # prefix cache: reuses the KV cache of what repeats every turn (system prompt + tool schemas), so
        # the time to the first token drops. It is on by default in vllm-mlx; explicit here on purpose.
        "--enable-prefix-cache",
    ]
    if cfg.get("CACHE_MEMORY_MB"):  # empty = automatic (~20% of the RAM)
        args += ["--cache-memory-mb", cfg["CACHE_MEMORY_MB"]]
    if (cfg.get("KV_CACHE_QUANTIZATION") or "").lower() == "true":  # saves memory on 24 GB
        args += [
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            cfg.get("KV_CACHE_QUANTIZATION_BITS") or "8",
        ]
    for key, flag in (
        ("DEFAULT_TEMPERATURE", "--default-temperature"),
        ("DEFAULT_TOP_P", "--default-top-p"),
        ("REASONING_PARSER", "--reasoning-parser"),
        ("CHAT_TEMPLATE_KWARGS", "--default-chat-template-kwargs"),
    ):
        if cfg.get(key):  # empty = the server default
            args += [flag, cfg[key]]
    return args


def _port_in_use(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


def main(cfg=None):
    """`mlx-agent-brain`: starts the model server in the foreground (Ctrl-C stops it). It refuses to listen
    beyond this Mac and to start a second server on a port that is already taken."""
    if cfg is None:
        from .agent import load_config  # noqa: PLC0415 - only the command needs it

        cfg = load_config()
    host, port = _host_port(cfg)
    if host not in _LOOPBACK:
        print(f"ERROR: SERVER_HOST={host} would expose the model to the network. Use 127.0.0.1.", file=sys.stderr)
        return 2
    if _port_in_use(host, port):
        print(
            f"ERROR: port {port} is already in use (is the brain already running?). "
            "Stop that process or change SERVER_PORT in config.env.",
            file=sys.stderr,
        )
        return 1
    exe = _vllm_mlx()
    if exe is None:
        print("ERROR: vllm-mlx is not installed. Run:  uv sync", file=sys.stderr)
        return 1
    args = server_args(cfg)
    print(f"Model server (vllm-mlx): {args[1]} as '{args[3]}' on http://{host}:{port} (loopback only)", flush=True)
    os.execv(exe, [exe, *args])  # noqa: S606 - fixed executable, arguments from config.env  # nosec B606
    return 0  # pragma: no cover: execv does not return


if __name__ == "__main__":
    raise SystemExit(main())
