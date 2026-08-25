"""src/brain.py: lifecycle of the BRAIN SERVER (vllm-mlx / Qwen).

Used by (1) the vision handoff (vision.read_image_auto): when 24 GB cannot hold the brain
and the image reader at the same time, the brain is PAUSED for a moment, the image is read, and the
brain is REACTIVATED. The 4 atomic operations live here; the orchestration (with the GOLDEN RULE: the
brain ALWAYS comes back) lives in vision.read_image_auto. And (2) the web STATUS PANEL:
`served_model` is its health-check with the observed truth (which model /v1/models serves).

The handoff does not run if there is enough RAM or if it is disabled by config.
"""
import json
import subprocess  # nosec B404: launches start_local.sh (fixed arg list, no shell)
import time
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:  # psutil is a project dependency; without it, stop() cannot locate the process
    psutil = None

ROOT = Path(__file__).resolve().parent.parent   # code lives in src/; start_local.sh in the root
_PATTERN = "vllm-mlx"        # marker of the brain server process in its cmdline


def _host_port(cfg):
    return (cfg.get("SERVER_HOST") or "127.0.0.1"), (str(cfg.get("SERVER_PORT") or "8000"))


def is_alive(cfg, timeout=4):
    """Does the brain server respond? (health-check via /v1/models). True/False, never raises."""
    host, port = _host_port(cfg)
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=timeout) as r:  # nosec B310: fixed loopback
            return r.status == 200
    except Exception:  # noqa: BLE001: any failure means not alive
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
    except Exception:  # noqa: BLE001: any failure means the brain is down/unresponsive
        return None
    names = [str(m.get("id") or "") for m in (data.get("data") or []) if isinstance(m, dict)]
    return ", ".join(n for n in names if n)      # "" = alive but with no models listed (rare; honest)


def _brain_pids():
    """PIDs of the brain server processes (vllm-mlx serve). [] if none or no psutil."""
    if psutil is None:
        return []
    pids = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if _PATTERN in cl and "serve" in cl:
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
        return True   # already stopped
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
    for pid in _brain_pids():   # the stubborn ones get a hard kill
        try:
            psutil.Process(pid).kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):  # noqa: BLE001
            continue
    time.sleep(2)   # settle: give the OS a moment to reclaim the model's RAM
    return not is_alive(cfg, timeout=2)


def start(cfg):
    """Starts the brain server (start_local.sh, which reads config.env for the correct model/flags) in
    a DETACHED process (start_new_session: survives the web process). Returns the Popen (or None on failure)."""
    script = ROOT / "start_local.sh"
    if not script.exists():
        return None
    try:
        return subprocess.Popen(  # nosec B603: fixed path within this repo, no shell
            [str(script)], cwd=str(ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except Exception:  # noqa: BLE001: if it does not start, the health-check detects it and reports honestly
        return None


def wait_for_health(cfg, timeout=150):
    """Waits (health-check) for the brain to respond again after starting. True if it came back in time."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_alive(cfg, timeout=3):
            return True
        time.sleep(2)
    return False
