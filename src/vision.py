"""src/vision.py: reads images with an ON-DEMAND VISION WORKER.

Additive path. When a chat turn carries an image, the backend calls `read_image()`; with no image,
none of this module runs and the normal path (Qwen brain) is untouched.

Why a SUBPROCESS instead of loading the model in the server process:
  - The worker is `python -m mlx_vlm.generate ...`: it loads the vision model, reads the image, prints
    the answer and exits. On exit the OS frees ALL of its RAM. That way a second model is never left
    resident permanently next to the brain (Qwen ~8 GB), so in 24 GB there are never two full models
    stuck together.
  - It is the simplest and safest option (the plan asked for "load on image, then release"). Cost: every
    image pays the model's cold start (~30-60 s). Acceptable for occasional vision use.

MEMORY GUARD: before starting the worker we measure free RAM; if it does not fit with margin, we degrade
honestly (with a warning) and never hang the Mac. The subprocess has a timeout: if it takes too long, it
is killed.

CONFIG-DRIVEN: the vision model is `VISION_MODEL` in config.env (default Gemma-4-12B); it can be switched
to `mlx-community/Qwen3-VL-8B-...` without touching code (mlx-vlm serves both). Known honest limit: Gemma
reads tables, logs, IPs, ports, codes and charts at 100%, but FAILS on long cryptic hashes/strings
(measured on the bench), documented in the design notes.
"""
import importlib.util
import re
import subprocess  # nosec B404: mlx_vlm.generate is invoked with a fixed arg list (no shell)
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # psutil is a project dependency; if missing, the RAM guard degrades to "unknown"
    psutil = None

# Extensions we treat as images (for detection; NOT in knowledge.DOC_EXTS, so they are never indexed).
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# mlx-vlm prints stats at the end; we use them for metrics and as an end-of-generation marker.
_RX_TOKS = re.compile(r"Generation:\s*\d+\s*tokens?,\s*([\d.]+)\s*tokens-per-sec")
_RX_PEAK = re.compile(r"Peak memory:\s*([\d.]+)\s*GB")
_END_MARK = "=========="   # mlx-vlm frames the generation between lines of '=' before the stats

# Conservative defaults (all adjustable via config.env).
_DEF_MODEL = "mlx-community/gemma-4-12b-it-4bit"
_DEF_MAX_TOKENS = 300
_DEF_MIN_FREE_GB = 9.0      # Gemma-4-12B 4-bit takes ~7.6 GB; we ask for margin to load without swap
_DEF_TIMEOUT_SEC = 300


def is_image(name):
    """Does the name/path point to an image (by extension)?"""
    return Path(str(name)).suffix.lower() in IMG_EXTS


def _num(cfg, key, default):
    try:
        v = (cfg.get(key) or "").strip()
        return type(default)(v) if v else default
    except (ValueError, TypeError):
        return default


def vision_model(cfg):
    """The vision model id (config-driven). Empty in config means the default (Gemma)."""
    return (cfg.get("VISION_MODEL") or _DEF_MODEL).strip()


def available(cfg):
    """(ok, detail). ok=False with an honest reason if vision cannot be used (not configured / no lib)."""
    if (cfg.get("VISION_MODEL") or "").strip() == "" and cfg.get("VISION_MODEL") is not None:
        # VISION_MODEL present but empty = vision deliberately disabled.
        return (False, "vision is disabled (VISION_MODEL empty in config.env).")
    if importlib.util.find_spec("mlx_vlm") is None:
        return (False, "mlx-vlm is not installed (uv pip install --python .venv/bin/python mlx-vlm).")
    return (True, vision_model(cfg))


def free_ram_gb():
    """Available RAM in GB (or None if it cannot be measured)."""
    if psutil is None:
        return None
    return psutil.virtual_memory().available / (1024 ** 3)


def _clean_output(txt, prompt):
    """mlx-vlm ECHOES the prompt and frames the generation with '=========='. Also, Gemma is a
    *thinking* model: mlx-vlm dumps the RAW template tokens and the thought channel
    (`<bos> <|turn>user <|image|> ... <turn|> <|turn>model <|channel>thought ... <channel|> ANSWER`).
    We extract ONLY the answer: after the prompt echo, after the thought channel closes and before
    the stats; then the residual special tokens are cleaned up."""
    gen = txt.split(prompt)[-1] if prompt and prompt in txt else txt
    # 1) cut mlx-vlm's trailing stats
    gen = gen.split(_END_MARK)[0] if _END_MARK in gen else gen
    # 2) the answer comes AFTER the last thought-channel close (drops reasoning + template)
    if "<channel|>" in gen:
        gen = gen.split("<channel|>")[-1]
    # 3) strip residual special tokens: <bos> <|turn> <turn|> <|image|> <|channel> <end_of_turn> ...
    gen = re.sub(r"<\|?[a-zA-Z0-9_]*\|?>", " ", gen)
    # 4) strip a stray marker word at the start (model/thought/assistant/final/analysis)
    gen = re.sub(r"^\s*(?:model|thought|assistant|final|analysis)\b", " ", gen, flags=re.IGNORECASE)
    # 5) filter out any mlx-vlm headers that survived
    lines = [ln for ln in gen.splitlines()
             if not ln.strip().startswith(("Files:", "Prompt:", "Image:", "==========", "Fetching"))]
    return "\n".join(lines).strip()


def read_image(img_path, prompt, cfg, log=None, project_dir=None):
    """Reads `img_path`, answering `prompt`, with the on-demand vision worker.

    Returns (text, meta:dict). NEVER raises on hang/RAM/timeout: it degrades honestly, returning a
    warning text and meta['ok']=False. `log(str)` optional = progress callback (for the WS).
    `project_dir` optional: if passed, the image is required to be CONFINED there (extra path safety).
    """
    def _notify(m):
        if log:
            try:
                log(m)
            except Exception:  # noqa: BLE001: the progress log must never take down vision
                pass

    path = Path(str(img_path)).resolve()
    meta = {"ok": False, "model": vision_model(cfg), "sec": 0.0, "tok_s": 0.0, "ram_peak_gb": 0.0}

    # 0) availability (config + lib)
    ok, detail = available(cfg)
    if not ok:
        return (f"I can't read the image: {detail}", meta)

    # 1) the file exists, is an image, and is confined to the project (path safety)
    if not path.is_file():
        return ("I can't find the image (was it uploaded to the project?).", meta)
    if not is_image(path):
        return (f"'{path.name}' does not look like an image ({', '.join(sorted(IMG_EXTS))}).", meta)
    if path.is_symlink():
        return ("Image path not allowed.", meta)
    if project_dir is not None:
        base = Path(str(project_dir)).resolve()
        if base not in path.parents:
            return ("Image path outside the project (not allowed).", meta)

    # 2) RAM GUARD: do not start the worker if it does not fit with margin (never hang the Mac).
    # SETTLE-RETRY: if a PREVIOUS vision worker just exited, the OS takes a moment to reclaim its pages
    # (on macOS `available` counts them out until then). If the 1st measurement falls short, we wait a
    # bit and re-measure ONCE before degrading; that way we don't reject over a transient dip.
    min_free = _num(cfg, "VISION_MIN_FREE_GB", _DEF_MIN_FREE_GB)
    free = free_ram_gb()
    if free is not None and free < min_free:
        _notify("memory is tight; waiting for it to be freed...")
        time.sleep(3)
        free = free_ram_gb()
    if free is not None and free < min_free:
        return (f"I can't read the image right now: not enough memory "
                f"(~{free:.1f} GB free and the vision reader needs ~{min_free:.0f} GB). "
                f"Close some apps or the model server and try again.", meta)

    model = vision_model(cfg)
    max_tokens = _num(cfg, "VISION_MAX_TOKENS", _DEF_MAX_TOKENS)
    timeout = _num(cfg, "VISION_TIMEOUT_SEC", _DEF_TIMEOUT_SEC)
    _notify(f"Reading the image with {model.split('/')[-1]} (cold start, ~30-60 s)...")

    # 3) subprocess worker: load, read, exit (frees RAM when done)
    cmd = [sys.executable, "-m", "mlx_vlm.generate",
           "--model", model,
           "--image", str(path),
           "--prompt", prompt,
           "--max-tokens", str(max_tokens),
           "--temperature", "0.0"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # nosec B603: fixed list, no shell
    except subprocess.TimeoutExpired:
        return (f"Reading the image took too long (>{timeout} s) and I cancelled it so as not to block the "
                f"Mac. Try a smaller image or try again.", meta)
    except Exception as e:  # noqa: BLE001: any worker failure is reported honestly, not propagated
        return (f"I couldn't read the image (vision reader failure: {type(e).__name__}).", meta)

    dt = time.time() - t0
    output = (r.stdout or "") + "\n" + (r.stderr or "")
    text = _clean_output(r.stdout or "", prompt)
    mt = _RX_TOKS.search(output)
    mp = _RX_PEAK.search(output)
    meta.update(sec=round(dt, 1),
                tok_s=float(mt.group(1)) if mt else 0.0,
                ram_peak_gb=float(mp.group(1)) if mp else 0.0)

    if r.returncode != 0 or not text:
        tail = (r.stderr or "").strip().splitlines()[-1:] or (r.stdout or "").strip().splitlines()[-1:]
        hint = (" . " + tail[0][:120]) if tail else ""
        return (f"I couldn't extract an answer from the image (the reader returned code {r.returncode}"
                f"{hint}).", meta)

    meta["ok"] = True
    return (text, meta)


def compose_vision_prompt(question, context_msgs, cfg):
    """Prepends to the question a BRIEF block from the chat thread (so Gemma understands references
    like "that thing above" and follow-up feels natural). Genuinely LIGHT context: config-gated by
    `VISION_CONTEXT` (default on) and capped at `VISION_CONTEXT_MAX_CHARS`.

    - No history (first turn) or the switch at 0 returns the question AS IS (nothing odd).
    - The context is ONLY to resolve references; the prompt makes clear the answer comes from the
      IMAGE (not to invent what the context mentions but the image does not show).

    `context_msgs` = list of [{role, content}] from the light assembly (summary + last N)."""
    on = str(cfg.get("VISION_CONTEXT", "1")).strip().lower() not in ("0", "false", "no", "off", "")
    if not on or not context_msgs:
        return question
    max_chars = _num(cfg, "VISION_CONTEXT_MAX_CHARS", 1200)
    lines = []
    for m in context_msgs:
        content = " ".join((m.get("content") or "").split())   # collapse spaces/newlines into a compact line
        if not content:
            continue
        label = {"user": "user", "assistant": "assistant"}.get(m.get("role"), "")
        lines.append(f"{label}: {content}" if label else content)
    block = "\n".join(lines).strip()
    if not block:
        return question
    if len(block) > max_chars:                                 # hard cap (light context, don't dump history)
        block = block[:max_chars].rstrip() + " ...[trimmed]"
    return ("Conversation context (ONLY to understand references like \"that\" or \"the earlier one\"; "
            "do NOT use it to invent what is not in the image):\n" + block +
            "\n\nUser question ABOUT THE IMAGE: " + question)


def read_image_auto(img_path, prompt, cfg, log=None, project_dir=None, brain=None):
    """Reads the image while MANAGING RAM automatically.

    - If there is enough RAM (or handoff is disabled): reads DIRECTLY (fast path, without touching the
      brain).
    - If there is NOT enough RAM and `VISION_AUTO_HANDOFF` is on and the brain is alive: does the HANDOFF,
      stops the brain, reads the image, and ALWAYS restarts the brain (golden rule: try/finally +
      health-check). Never leaves the user without chat, never hangs.

    `brain` = module/object with is_alive/stop/start/wait_for_health (injectable for tests; by default
    imports `brain`). Returns (text, meta) like read_image. Never propagates exceptions."""
    def _log(m):
        if log:
            try:
                log(m)
            except Exception:  # noqa: BLE001: the progress log must never take down the read
                pass

    min_free = _num(cfg, "VISION_MIN_FREE_GB", _DEF_MIN_FREE_GB)
    auto = str(cfg.get("VISION_AUTO_HANDOFF", "1")).strip().lower() not in ("0", "false", "no", "off", "")
    free = free_ram_gb()

    # FAST PATH: enough / unknown RAM / handoff off, read directly (without touching the brain).
    if free is None or free >= min_free or not auto:
        return read_image(img_path, prompt, cfg, log=log, project_dir=project_dir)

    # RAM is tight + handoff on. Is there a live brain using RAM that we can pause?
    if brain is None:
        import brain  # noqa: PLC0415: lazy import (tests inject a double)
    if not brain.is_alive(cfg):
        return read_image(img_path, prompt, cfg, log=log, project_dir=project_dir)  # nothing to hand off

    # HANDOFF
    _log(f"Low free RAM (~{free:.1f} GB): pausing the brain for a moment to read the image "
         f"(this will take a bit longer than usual)...")
    if not brain.stop(cfg):
        # couldn't pause the brain, it is still alive (chat is safe); we don't start a 2nd. Let the guard decide.
        _log("Couldn't free memory by pausing the brain; trying to read if it fits.")
        return read_image(img_path, prompt, cfg, log=log, project_dir=project_dir)
    try:
        return read_image(img_path, prompt, cfg, log=log, project_dir=project_dir)
    except Exception as e:  # noqa: BLE001: NEVER propagate: return honestly (the finally restarts the brain)
        return (f"I couldn't read the image ({type(e).__name__}).",
                {"ok": False, "model": vision_model(cfg), "sec": 0.0, "tok_s": 0.0, "ram_peak_gb": 0.0})
    finally:
        # GOLDEN RULE (first thing): whatever happens, success, failure, timeout, the brain COMES BACK.
        _log("Restarting the brain...")
        brain.start(cfg)
        if brain.wait_for_health(cfg):
            _log("Brain is back. You can keep chatting.")
        else:
            _log("The brain is taking a while to come back; give it a few seconds and retry the chat.")


# --- Direct smoke:  python src/vision.py <image> "<question>"  ---
if __name__ == "__main__":  # pragma: no cover: manual test utility
    import json
    if len(sys.argv) < 3:
        print("usage: python src/vision.py <image_path> \"<question>\"", file=sys.stderr)
        sys.exit(2)
    _cfg = {"VISION_MODEL": _DEF_MODEL}
    _txt, _meta = read_image(sys.argv[1], sys.argv[2], _cfg, log=lambda m: print(m, file=sys.stderr))
    print("\n----- ANSWER -----")
    print(_txt)
    print("----- META -----")
    print(json.dumps(_meta, ensure_ascii=False))
