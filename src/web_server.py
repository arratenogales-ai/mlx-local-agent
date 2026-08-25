#!/usr/bin/env python3
# web_server.py: thin web backend over the agent (Level 5A, visual chat).
#
# A thin layer over the existing system: it does NOT rewrite the agent's logic.
# It only does three things:
#   1) serves the chat frontend (web/index.html),
#   2) receives a task over WebSocket,
#   3) runs `resolve_task` (the usual orchestrator) in a thread and, via the
#      event hook (events.use_emitter), forwards LIVE to the web each
#      plan / step / tool / verification / verdict / memory / final.
#
# The agent doesn't know the web exists: it emits events to the hook, that's all.
# Without the web the hook is a no-op and the CLI works identically (clean separation).
#
# 100% local: runs on localhost against the Level 1 MLX server.
#
# Usage:  ./web.sh   (equivalent to running uvicorn via this file)
import asyncio
import os
import subprocess  # nosec B404: only `git rev-parse` (fixed arg list, no shell) for the version (N17)
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import agent as _agent
from agent import load_config
from orchestrator import (build_router, resolve_task, resolve_in_project,
                          preview_undo, apply_undo, _set_workdir)
from events import use_emitter
import projects  # Level 7B: project selector (thin layer over the 7A backend)
import chats      # Level 14: chats inside projects (migration + per-chat conversation)
import voice      # Level 8D: local voice (TTS with `say`; STT with mlx-whisper if available)
import skills  # Level 9C: skills catalog for the panel (and forcing one by hand)
import knowledge  # Level 10C: (re)index a file uploaded to the project
import vision        # Phase 2: on-demand vision worker (mlx-vlm) for attached images
import worker         # Level 15: worker mode (agentic) in a separate tab from the chat
import brain        # N17 Phase A: brain health + served model (for the status panel)
import web_tools  # N17 Phase A: the SAME WEB_SEARCH gate (fail-closed) the tools use

ROOT = Path(__file__).resolve().parent.parent   # code lives in src/; web/ (and config.env) at the root
WEB = ROOT / "web"

app = FastAPI(title="Local agent, visual chat (Level 5A)")
_stt_lock = asyncio.Lock()      # 8D: serializes transcriptions (never two at once)
# WORKDIR is process-GLOBAL state (resolve_in_project sets it to the project): we serialize the agent's
# execution so two concurrent web requests (two tabs/projects) don't clobber each other.
_agent_lock = threading.Lock()

# Config and router are built ONCE at startup. Local by default (STRONG empty); the agnostic router picks
# each role's brain without touching anything here.
cfg = load_config()


def _git_version():
    """N17 Phase A: git version of the code THIS process serves (computed once at startup: the version
    loaded in memory, not the one on disk). `git describe --always --dirty` rather than just the hash:
    with uncommitted changes the label gets a "-dirty" suffix, otherwise two servers running different
    code could show the SAME label (a false equality signal, adversarial N17-A review). Returns '' if
    there's no git/repo (the web shows "unknown"); never breaks startup."""
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"],  # nosec B603 B607: fixed args, no shell
                           cwd=str(ROOT), capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001: no git / timeout -> unknown version (honest degradation)
        return ""


GIT_VERSION = _git_version()

# N17 Phase A: is a VISION turn in flight? The RAM handoff pauses the brain ON PURPOSE while reading an
# image (tens of seconds, up to minutes on a cold start): the status panel uses this to show "vision
# handoff" in amber instead of shouting "brain down, restart it" (which would push the user to start a
# 2nd model with RAM already full, exactly the OOM the handoff avoids).
_vision_in_progress = False


async def _json_body(request):
    """Read the JSON body safely: returns the dict, or None if the body is invalid/not an object
    (so a POST with garbage returns a clean 400 instead of an unhandled 500)."""
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001: non-JSON / empty body
        return None
    return d if isinstance(d, dict) else None


def _stt_limit_bytes():
    try:
        return max(1, int(cfg.get("STT_MAX_MB") or 25)) * 1024 * 1024   # audio upload limit
    except (TypeError, ValueError):
        return 25 * 1024 * 1024


def _upload_limit_bytes():
    try:
        return max(1, int(cfg.get("UPLOAD_MAX_MB") or 20)) * 1024 * 1024  # dropped-file limit (10C)
    except (TypeError, ValueError):
        return 20 * 1024 * 1024


UPLOAD_TIMEOUT = 60        # time limit to receive the body (a slow client must not hang the worker)
INDEX_TIMEOUT = 30         # limit to (re)index in /upload (if slow, it's saved anyway and indexed on search)


async def _read_bounded_upload(upload: UploadFile, limit: int):
    """Read an UploadFile in chunks, capping by SIZE (limit -> 413) and by TIME (deadline -> 408).
    Returns (bytearray, None) or (None, error JSONResponse). Anti-hang: a slow client can't keep a worker
    busy indefinitely."""
    data = bytearray()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + UPLOAD_TIMEOUT
    while True:
        try:
            chunk = await asyncio.wait_for(upload.read(1 << 16), timeout=max(0.1, deadline - loop.time()))
        except asyncio.TimeoutError:
            return None, JSONResponse({"error": "upload too slow (timeout)"}, status_code=408)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            return None, JSONResponse({"error": "file too large"}, status_code=413)
    return data, None


# N18 B2: LAZY INIT of the router. It used to be built HERE, at import time, and `build_router` calls
# `sys.exit` if the brain (8000) doesn't respond, so importing `web_server` (which many tests and the
# verify.sh helper do) ABORTED the whole suite with the brain off (the N17 false-red). Now the router
# is built on the FIRST turn that needs it (cached, thread-safe), and `build_client`'s `sys.exit` is turned
# into a NORMAL exception so the handler reports it honestly (the health panel already warns) instead of
# killing a server thread. Importing the module no longer touches the brain.
_router = None
_router_lock = threading.Lock()


def _get_router():
    """The model router, built LAZILY the first time (not at import). If the brain doesn't respond,
    `build_router`->`build_client` would `sys.exit`; we catch it and re-raise as RuntimeError so the
    chat/worker loop gives an honest message (never a false success, never a hang)."""
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                try:
                    _router = build_router(cfg)
                except SystemExit as e:   # build_client calls sys.exit if the model doesn't respond
                    raise RuntimeError(f"the brain (model) is not responding on 8000 ({e}). Start it with ./start_local.sh") from e
    return _router


# Respect PROJECT_MEMORY from config.env just like the CLI (the web used to always force it on).
use_mem = cfg.get("PROJECT_MEMORY", "true").strip().lower() == "true"

# Frontend static files (CSS/JS if any). The index is served separately.
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/")
def index():
    """The chat page (a single HTML with embedded CSS/JS, all local). `no-store` so the browser does NOT
    serve a stale cached JS version (which caused the mic to send the old format)."""
    return FileResponse(str(WEB / "index.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/agentic")
def agentic_view():
    """Level 15: the WORKER MODE (agentic) view, in a SEPARATE tab from the chat. Additive: the chat (`/`)
    doesn't change. Its own HTML with embedded CSS/JS (Aurora theme), all local."""
    return FileResponse(str(WEB / "agentic.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.post("/worker-control")
async def api_worker_control(request: Request):
    """Level 15 Phase 3 + N16 Phase C: worker control: {project, signal}. Signals: pause|continue|cancel
    (between passes) and approve|redo (between STAGES, human-in-the-loop). The signal is persisted in
    the project (.worker/control.json) and the worker reads it BETWEEN passes/stages. Anti-CSWSH + validation."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    project = (data.get("project") or "").strip()
    signal_ = (data.get("signal") or "").strip()
    if not project or project == "undefined" or not worker.project_dir(cfg, project).is_dir():
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    if signal_ not in ("pause", "continue", "cancel", "approve", "redo"):   # +Phase C: approve/redo stage
        return JSONResponse({"error": "invalid signal (pause|continue|cancel|approve|redo)"}, status_code=400)
    worker.set_signal(cfg, project, signal_)
    return {"ok": True, "signal": signal_}


@app.get("/worker-status")
def api_worker_status(project: str = ""):
    """Level 15 Phase 3: persistent worker state in a project (to resume/read in the morning, and to
    RESTORE the last run in the view on reload). Also accepts manually created folders
    (worker.project_dir, confined to the base)."""
    project = (project or "").strip()
    if not project or project == "undefined" or not worker.project_dir(cfg, project).is_dir():
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    return {"state": worker.read_state(cfg, project), "signal": worker.read_signal(cfg, project)}


def _origin_ok(origin):
    """Only the local page itself (same anti-CSWSH as the WebSocket). No Origin (script) -> OK."""
    return origin is None or (urlparse(origin).hostname or "") in ("127.0.0.1", "localhost")


@app.get("/system-status")
def api_system_status(request: Request):
    """N17 Phase A: system state at a glance (web panel): is the brain (MLX server) responding and which
    model does it ACTUALLY serve? (brain.served_model = truth observed via /v1/models, not config), is web
    search enabled? (the same fail-closed gate the tools use), and the git version of the code serving this
    web. `web` is trivially True (if we answer, the web is alive); the frontend renders the row anyway and
    treats a fetch FAILURE as "web down". Anti-CSWSH."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    model = brain.served_model(cfg, timeout=2)      # None = down; "" = alive with no models (rare)
    return {
        "web": True,
        "version": GIT_VERSION,
        "brain": model is not None,
        "model": model or "",
        "model_config": (cfg.get("MLX_MODEL") or "").rsplit("/", 1)[-1],   # what it's configured with
        "web_search": web_tools._search_enabled(),              # noqa: SLF001: canonical N16 gate
        "vision_in_progress": _vision_in_progress,          # vision handoff != brain down (honesty)
    }


# --- Level 14: chat helpers (lazy migration + active chat folder) ---
def _web_chat_dir(project, chat=""):
    """CHAT folder (active, or the given sanitized one) of a project. Migrates the project lazily (with
    backup) if it was still on the old model. Returns the chat folder (or the project itself if there are
    no chats, for compat)."""
    proj = projects.project_path(cfg, project)
    chats.ensure_chat(proj, base=projects.base_dir(cfg))
    cid = chats._slug(chat) if (chat or "").strip() else chats.active_chat(proj)
    return chats.chat_path(proj, cid) if cid else proj


def _project_memory(project):
    """Project memory (.memory.md, shared), bounded. '' if none."""
    p = projects.project_path(cfg, project) / ".memory.md"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()[:2000]
    except (OSError, UnicodeError):
        return ""


def _respond_image(project, chat, image, prompt, emitter):
    """N17 Phase A: thin wrapper around the vision path: sets `_vision_in_progress` (try/finally, always
    cleared) so /system-status can tell "brain paused by the handoff" from "brain down"."""
    global _vision_in_progress
    _vision_in_progress = True
    try:
        return _respond_image_impl(project, chat, image, prompt, emitter)
    finally:
        _vision_in_progress = False


def _respond_image_impl(project, chat, image, prompt, emitter):
    """Phase 2: VISION PATH (additive): a turn brings an attached image (already uploaded to the project
    via /upload) -> the on-demand VISION WORKER reads it (vision.read_image_auto: an mlx-vlm subprocess
    that loads and frees; RAM guard; degrades honestly). Does NOT touch WORKDIR or the normal path.
    Persists the turn in the chat conversation so history/search/export stay coherent."""
    if not project:
        return "Open or create a project and upload the image there before asking about it."
    proj = projects.project_path(cfg, project)           # sanitized name (anti-traversal)
    name = Path(image).name                               # PATH-SAFETY: only the name, never a path
    img_path = (proj / name).resolve()
    question = (prompt or "").strip() or "Describe the image and extract any relevant data (text, figures, tables)."

    # Phase 2.2: LIGHT CONVERSATIONAL CONTEXT: prepend to Gemma's prompt the summary + last N of the active
    # chat (same assembly as Level 14, bounded) so it understands references ("that thing above").
    # Config-gated (VISION_CONTEXT) and capped (VISION_CONTEXT_MAX_CHARS). First turn / no history ->
    # empty context -> prompt as-is. chat_dir is computed once (reused to persist).
    chat_dir = None
    ctx_msgs = []
    try:
        chat_dir = _web_chat_dir(project, chat)
        if str(cfg.get("VISION_CONTEXT", "1")).strip().lower() not in ("0", "false", "no", "off", ""):
            n = int(cfg.get("CONV_LAST_N") or 6)
            ctx_msgs = projects.conversation_context(chat_dir, n)
    except Exception:  # noqa: BLE001: if context fails, read the image WITHOUT context (never breaks vision)
        ctx_msgs = []
    vision_prompt = vision.compose_vision_prompt(question, ctx_msgs, cfg)

    text, meta = vision.read_image_auto(                 # Phase 2.1: manages RAM (brain handoff if needed)
        img_path, vision_prompt, cfg,
        log=lambda m: emitter({"type": "vision", "text": m}),
        project_dir=proj)                               # confines the read to the project (extra path-safety)
    try:                                                  # persist the turn (the ORIGINAL question, not the augmented prompt)
        if chat_dir is None:
            chat_dir = _web_chat_dir(project, chat)
        projects.save_turn(chat_dir, f"🖼️ [{name}] {question}", text)
    except Exception:  # noqa: BLE001: a failure saving history must never lose the answer already obtained
        pass
    return text


# --- Level 7B: projects API (thin layer over projects.py / the 7A backend) ---
@app.get("/projects")
def api_list_projects():
    return {"projects": projects.list_projects(cfg), "active": projects.active_project(cfg)}


@app.post("/projects")
async def api_create_project(request: Request):
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    name = (data.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "empty name"}, status_code=400)
    projects.create_project(cfg, name)
    projects.set_active(cfg, name)
    return {"projects": projects.list_projects(cfg), "active": projects.active_project(cfg)}


@app.post("/projects/delete")
async def api_delete_project(request: Request):
    """Delete a project (to the .trash/ trash). Destructive -> anti-CSWSH + backend validation
    (sanitized name, only inside the base). The frontend asks for explicit confirmation."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    name = (data.get("name") or "").strip()
    if not name:                               # empty name -> delete NOTHING (avoids deleting 'project')
        return JSONResponse({"error": "empty name"}, status_code=400)
    res = projects.delete_project(cfg, name)
    if not res.get("ok"):
        return JSONResponse(res, status_code=404)
    return {"ok": True, "projects": projects.list_projects(cfg),
            "active": projects.active_project(cfg)}


@app.post("/active-project")
async def api_activate_project(request: Request):
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    name = (data.get("name") or "").strip()
    if not projects.exists(cfg, name):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    requested_chat = (data.get("chat") or "").strip()          # optional: open a specific chat
    projects.set_active(cfg, name)
    proj = projects.project_path(cfg, name)
    # Level 14: migrate (lazy, with backup) and return the chat list + active chat + ITS conversation + the
    # PROJECT memory (shared) + the CHAT memory (summary). Light context: only the last messages to RENDER,
    # never the whole history.
    chats.ensure_chat(proj, base=projects.base_dir(cfg))
    cid = chats._slug(requested_chat) if requested_chat else chats.active_chat(proj)
    if not cid or cid not in chats.list_chats(proj):
        cid = chats.active_chat(proj)
    chats.set_active_chat(proj, cid)
    chat_dir = chats.chat_path(proj, cid)
    conv = projects.load_conversation(chat_dir)[-40:]
    return {"active": projects.active_project(cfg), "conversation": conv,
            "chats": chats.list_chats(proj), "active_chat": cid,
            "project_memory": _project_memory(name), "chat_memory": projects.read_summary(chat_dir)}


# --- Level 14: chats API inside a project (anti-CSWSH + path-safety + caps) ---
_MAX_CHAT_NAME = 64


@app.get("/chats")
def api_list_chats(request: Request, project: str = ""):
    """List a project's chats + the active one (migrate lazily if needed). Anti-CSWSH."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    project = (project or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    proj = projects.project_path(cfg, project)
    chats.ensure_chat(proj, base=projects.base_dir(cfg))
    return {"chats": chats.list_chats(proj), "active": chats.active_chat(proj)}


@app.post("/chats")
async def api_create_chat(request: Request):
    """Create a chat in a project and make it active. Anti-CSWSH + path-safety (slug) + name cap."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    project = (data.get("project") or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    name = (data.get("name") or "").strip()[:_MAX_CHAT_NAME]   # size cap
    proj = projects.project_path(cfg, project)
    chats.ensure_chat(proj, base=projects.base_dir(cfg))
    new_chat = chats.create_chat(proj, name or None)             # path-safe slug; no name -> chat-N
    return {"ok": True, "chats": chats.list_chats(proj), "active": new_chat}


@app.post("/active-chat")
async def api_activate_chat(request: Request):
    """Switch a project's active chat and return ITS conversation + chat memory. Anti-CSWSH."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    project = (data.get("project") or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    proj = projects.project_path(cfg, project)
    chats.ensure_chat(proj, base=projects.base_dir(cfg))
    cid = chats._slug((data.get("chat") or "").strip())
    if not cid or cid not in chats.list_chats(proj):
        return JSONResponse({"error": "chat does not exist"}, status_code=404)
    chats.set_active_chat(proj, cid)
    chat_dir = chats.chat_path(proj, cid)
    return {"ok": True, "active_chat": cid, "conversation": projects.load_conversation(chat_dir)[-40:],
            "chat_memory": projects.read_summary(chat_dir)}


@app.post("/chats/delete")
async def api_delete_chat(request: Request):
    """Delete a chat (to the project trash). Destructive -> anti-CSWSH + backend path-safety.
    The frontend asks for explicit confirmation."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    project = (data.get("project") or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    chat = (data.get("chat") or "").strip()
    if not chat:
        return JSONResponse({"error": "empty chat"}, status_code=400)
    proj = projects.project_path(cfg, project)
    res = chats.delete_chat(proj, chat)
    if not res.get("ok"):
        return JSONResponse(res, status_code=404)
    return {"ok": True, "chats": chats.list_chats(proj), "active": chats.active_chat(proj)}


# --- Level 8A: time machine (undo the last task) ---
@app.get("/undo-preview")
def api_undo_preview(project: str):
    """What undo would revert (to confirm in the web), without applying it."""
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    return preview_undo(cfg, project) or {"empty": True}


@app.post("/undo")
async def api_undo(request: Request):
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    project = (data.get("project") or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)

    # N17 Phase B: undo REVERTS/deletes/recreates project files: it must be SERIALIZED with the agent
    # (_agent_lock, like chat/worker/vision). Without the lock, an "undo" while the worker writes in the
    # SAME project would restore underneath it -> a mixed tree (half task, half snapshot). In a threadpool
    # so the event loop isn't blocked if the agent is mid long task.
    def _undo_serialized():
        with _agent_lock:
            return apply_undo(cfg, project)

    plan = await run_in_threadpool(_undo_serialized)
    return {"undone": plan is not None, "plan": plan}


# --- Level 8D: local voice (TTS with `say`; STT with mlx-whisper if available) ---
@app.get("/voice-status")
def api_voice_status():
    def _num(key, default):
        try:
            return max(1, int(cfg.get(key) or default))
        except (TypeError, ValueError):
            return default
    return {"tts": voice.tts_available(), "stt": voice.stt_available(),
            "interval": _num("STT_INTERVAL_SEC", 3), "window": _num("STT_WINDOW_SEC", 30)}


@app.get("/skills")
def api_skills(request: Request):
    """Level 9C: a LIGHT skills catalog for the web panel (name + short description), including the active
    project's (9D) if it has its own folder. Empty if the subsystem is off. Read-only GET, same anti-CSWSH
    as the rest."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    active = projects.active_project(cfg)                        # 9D: active project's skills
    proj_dir = str(projects.project_path(cfg, active)) if active else None
    return {"skills": skills.web_catalog(cfg, project_dir=proj_dir)}


@app.post("/tts")
async def api_tts(request: Request):
    """Text -> WAV audio (100% local with `say`). Anti-CSWSH (a third-party site must not invoke it)."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    data = await _json_body(request)
    if data is None:
        return JSONResponse({"error": "invalid body"}, status_code=400)
    wav = voice.synthesize((data.get("text") or "").strip())
    if wav is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)
    return Response(content=wav, media_type="audio/wav")


@app.post("/stt")
async def api_stt(request: Request, audio: UploadFile = File(...)):
    """Audio -> text (speech-to-text) with mlx-whisper if installed; otherwise degraded. Anti-CSWSH +
    SIZE CAP (a huge audio must not blow up memory). SERIALIZES with `_stt_lock` (never two at once) and
    transcribes in a thread (run_in_threadpool) so the event loop isn't blocked."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    if not voice.stt_available():
        return JSONResponse({"degraded": True,
                             "error": "Dictation not available: install mlx-whisper (uv pip install mlx-whisper)."},
                            status_code=503)
    limit = _stt_limit_bytes()
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:      # fast reject by header (without reading)
        return JSONResponse({"error": "audio too large"}, status_code=413)
    data, err = await _read_bounded_upload(audio, limit)   # size + TIME (anti-hang for a slow client)
    if err is not None:
        return err
    try:
        async with _stt_lock:                        # one at a time; `async with` RELEASES the lock
            text = await run_in_threadpool(voice.transcribe, bytes(data), cfg.get("WHISPER_MODEL"))
    except Exception:  # noqa: BLE001: even if transcription blows up, the lock is freed; clean response
        return JSONResponse({"degraded": True, "error": "Could not transcribe the audio."}, status_code=503)
    if text is None:
        return JSONResponse({"degraded": True, "error": "Could not transcribe the audio."}, status_code=503)
    return {"text": text}


@app.post("/upload")
async def api_upload(request: Request, file: UploadFile = File(...), project: str = Form("")):
    """Level 10C: drop a file in the chat -> it's SAVED in the active project and, if it's an indexable
    document, (re)INDEXED. Anti-CSWSH + SIZE CAP (like /stt) + path-safety (only the name, no path;
    confined to the project). Requires a project (that's where its documents live)."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    project = (project or "").strip()
    if not project:
        return JSONResponse({"error": "open or create a project before uploading files"}, status_code=400)
    name = Path(file.filename or "").name          # PATH-SAFETY: only the name, never a path
    if not name or name.startswith("."):
        return JSONResponse({"error": "invalid file name"}, status_code=400)
    limit = _upload_limit_bytes()
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:       # fast reject by header
        return JSONResponse({"error": "file too large"}, status_code=413)
    data, err = await _read_bounded_upload(file, limit)   # size + TIME (anti-hang)
    if err is not None:
        return err
    proj = projects.project_path(cfg, project)       # name sanitized with _slug (anti-traversal)
    dest = (proj / name).resolve()
    if proj.resolve() not in dest.parents:            # extra defense: never outside the project
        return JSONResponse({"error": "path not allowed"}, status_code=400)
    if dest.is_symlink():                             # NEVER write through a symlink (anti-escape)
        return JSONResponse({"error": "path not allowed"}, status_code=400)
    replaced = dest.exists()
    try:
        proj.mkdir(parents=True, exist_ok=True)
        # ATOMIC write without following symlinks: mkstemp (O_EXCL, doesn't follow links) + os.replace
        # (replaces the destination without writing through a symlink created in a race window).
        fd, tmp_path = tempfile.mkstemp(dir=str(proj), prefix=".upload_", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes(data))
        os.replace(tmp_path, dest)
    except OSError as e:
        try:
            os.unlink(tmp_path)
        except (OSError, NameError):
            pass
        return JSONResponse({"error": f"could not save: {e}"}, status_code=500)
    indexed = False                                   # (re)index only if it's an indexable document
    if dest.suffix.lower() in knowledge.DOC_EXTS:
        try:                                          # TIMEOUT: if slow, it was saved anyway and indexed on search
            await asyncio.wait_for(run_in_threadpool(knowledge.index_docs, proj), timeout=INDEX_TIMEOUT)
            indexed = True
        except Exception:  # noqa: BLE001: an index failure/timeout (incl. TimeoutError) must not sink the upload
            indexed = False
    return {"ok": True, "name": name, "bytes": len(data), "indexed": indexed, "replaced": replaced}


# --- Level 12C: conversation search / export for the project (brings the CLI's 10D to the web) ---
_MAX_Q_CONV = 200                    # search query cap (anti-DoS; light context)
_MAX_EXPORT_CHARS = 2_000_000        # .md export cap (light context / output anti-DoS)
_EXPORT_FORMATS = {"md": (".md", "text/markdown; charset=utf-8"), "pdf": (".pdf", "application/pdf")}


@app.get("/conversation/search")
def api_search_conversation(request: Request, project: str = "", q: str = "", chat: str = ""):
    """Search the project's ACTIVE CHAT conversation (10D + 14) for turns matching `q`. BOUNDED results
    (light context: <=MAX_TURNS turns, each message trimmed). Anti-CSWSH: only the local page can read the
    conversation (a foreign Origin -> 403). Reuses conversations.py."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    project = (project or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    query = (q or "").strip()[:_MAX_Q_CONV]           # query size cap
    if not query:
        return {"results": [], "total": 0, "query": ""}
    import conversations
    res = conversations.search(_web_chat_dir(project, chat), query)   # 14: over the active chat
    return {"results": res, "total": len(res), "query": query}


@app.get("/conversation/export")
def api_export_conversation(request: Request, project: str = "", format: str = "md", chat: str = ""):
    """Export the project's ACTIVE CHAT conversation (10D + 14) as a .md or .pdf DOWNLOAD. Anti-CSWSH;
    degrades honestly if pandoc/LaTeX are missing (PDF -> 503 with reason). Reuses conversations.py."""
    if not _origin_ok(request.headers.get("origin")):
        return JSONResponse({"error": "origin not allowed"}, status_code=403)
    project = (project or "").strip()
    if not projects.exists(cfg, project):
        return JSONResponse({"error": "project does not exist"}, status_code=404)
    format = (format or "md").strip().lower()
    if format not in _EXPORT_FORMATS:
        return JSONResponse({"error": "unsupported format (use md or pdf)"}, status_code=400)
    import conversations
    proj = _web_chat_dir(project, chat)                 # 14: the active chat's conversation
    if not conversations.turns(proj):
        return JSONResponse({"error": "no conversation to export"}, status_code=404)
    ext, media = _EXPORT_FORMATS[format]
    cid = projects._slug(chat) if (chat or "").strip() else chats.active_chat(projects.project_path(cfg, project))
    download_name = f"conversation-{projects._slug(project)}-{cid or 'chat'}{ext}"
    if format == "md":
        md = conversations._to_markdown(proj)
        if len(md) > _MAX_EXPORT_CHARS:
            md = md[:_MAX_EXPORT_CHARS] + "\n\n...[truncated: very long conversation]"
        return Response(content=md, media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="{download_name}"'})
    # PDF: export to an ISOLATED temp file, serve it as a download and clean it up afterward (BackgroundTask).
    import os
    import tempfile
    from starlette.background import BackgroundTask
    tmpdir = tempfile.mkdtemp(prefix="conv_export_")
    tmp = Path(tmpdir) / download_name
    msg = conversations.export(proj, tmp)                # SAFE pandoc (temp + cleans destination on failure)
    if not (tmp.exists() and str(msg).startswith("OK")):
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
        return JSONResponse({"error": msg or "could not export to PDF (missing pandoc/LaTeX?)"}, status_code=503)

    def _cleanup():
        try:
            os.remove(tmp)
            os.rmdir(tmpdir)
        except OSError:
            pass
    return FileResponse(str(tmp), media_type=media, filename=download_name, background=BackgroundTask(_cleanup))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Live channel: the client sends {task, fast?}; the server runs the agent and forwards events as
    they happen, up to the final result."""
    # Security (5A): only accept connections from the local page itself. The browser attaches the Origin
    # header; a THIRD-PARTY site open in the same browser could connect to ws://127.0.0.1 and issue tasks
    # (CSWSH / "confused deputy"), and the agent runs shell. Binding to localhost does NOT protect against
    # that (the attacker is the browser, not the network). Non-browser clients (no Origin, e.g. a test
    # script) are allowed.
    origin = websocket.headers.get("origin")
    if origin is not None and (urlparse(origin).hostname or "") not in ("127.0.0.1", "localhost"):
        await websocket.close(code=1008)  # 1008 = policy violation
        return
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:  # noqa: BLE001,S112: non-JSON frame: ignore it and keep listening (don't break the WS)  # nosec B112
                continue
            if not isinstance(msg, dict):       # valid JSON but not an object (array/string) -> ignore
                continue
            task = (msg.get("task") or "").strip()
            fast = bool(msg.get("fast"))
            project = (msg.get("project") or "").strip()  # Level 7B: active project (or empty)
            chat = (msg.get("chat") or "").strip()          # Level 14: active chat (or empty -> the active one)
            forced_skill = (msg.get("skill") or "").strip() or None  # Level 9C: manually forced skill
            image = (msg.get("image") or "").strip()      # Phase 2: image attached to the turn (vision path)
            if not task and not image:
                continue
            if image and not task:                          # image with no question -> default description
                task = "Describe the image and extract any relevant data."

            # Queue linking the agent THREAD (event producer) with this async handler (consumer that sends
            # them over the WebSocket).
            queue: asyncio.Queue = asyncio.Queue()
            END = {"type": "__end__"}

            def emitter(ev):
                # Called FROM the agent thread: enqueue thread-safely.
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            def work():
                # Runs the usual orchestrator, with the hook active only here. With an active project (7B)
                # it runs INSIDE it (its folder/memory/persistent conversation); without one, usual behavior
                # (cwd, no conversation) -> backward compatibility.
                try:
                    # Phase 2: VISION PATH (additive): if the turn brings an image, the vision worker reads
                    # it and we return its answer. No image -> EXACTLY the usual path (block below).
                    # Phase 2.1: vision takes the _agent_lock, so it SERIALIZES with the chat (never two
                    # models at once) and, if the RAM HANDOFF is needed, the brain is NEVER paused with a
                    # chat in flight.
                    if image:
                        with _agent_lock:
                            resp = _respond_image(project, chat, image, task, emitter)
                        emitter({"type": "final", "text": resp})
                        return
                    # WORKDIR/CHATDIR are process globals. We capture and restore INSIDE the lock: if captured
                    # OUTSIDE, two concurrent web tasks could restore a WORKDIR already mutated by the other
                    # and leave it "stuck" to a project -> a later task WITHOUT a project would write into
                    # that project's shared .memory.md/.lessons.md (Level 14 review).
                    with _agent_lock:          # one agent task at a time (don't clobber WORKDIR between tabs)
                        wd_orig = _agent.WORKDIR   # captured inside the lock -> never contaminated by another task
                        try:
                            with use_emitter(emitter):
                                if project:
                                    resp = resolve_in_project(_get_router(), cfg, project, task, fast=fast,
                                                              forced_skill=forced_skill, chat=chat)
                                else:
                                    resp = resolve_task(_get_router(), cfg, task, fast=fast,
                                                        use_memory=(use_mem and not fast),
                                                        forced_skill=forced_skill)
                        finally:
                            _set_workdir(wd_orig)   # undo WORKDIR/CHATDIR INSIDE the lock (for the next one)
                    emitter({"type": "final", "text": resp})
                except Exception as e:  # noqa: BLE001: clean error to the web
                    emitter({"type": "error", "text": f"{type(e).__name__}: {e}"})
                finally:
                    emitter(END)

            # Immediate echo of the start (guaranteed order, before starting the thread).
            await websocket.send_json({"type": "start", "task": task, "fast": fast})
            fut = loop.run_in_executor(None, work)

            # ALWAYS drain the queue up to the __end__ sentinel, even if the client left: this way the agent
            # thread finishes cleanly and leaves no worker/queue dangling (the thread isn't cancelable: it's
            # the agent's logic, untouched). If sending fails (client disconnected), we stop sending but keep
            # draining.
            send_ok = True
            while True:
                ev = await queue.get()
                if ev.get("type") == "__end__":
                    break
                if send_ok:
                    try:
                        await websocket.send_json(ev)
                    except Exception:  # noqa: BLE001: the client left: drain but don't send
                        send_ok = False
            await fut  # reclaims the worker and propagates any unexpected exception from the thread
            if not send_ok:
                break  # the socket is dead: exit the receive loop
    except WebSocketDisconnect:
        pass  # the user closed the tab: normal end


@app.websocket("/ws-agentic")
async def ws_agentic(websocket: WebSocket):
    """Level 15: live channel for WORKER MODE (agentic). SEPARATE from the chat (`/ws`), additive: same
    pattern (thread-safe queue + thread + drain) but calls the `worker` instead of the chat orchestrator.
    Receives {task, project} and emits the worker's steps live (the `worker_*` protocol)."""
    origin = websocket.headers.get("origin")
    if origin is not None and (urlparse(origin).hostname or "") not in ("127.0.0.1", "localhost"):
        await websocket.close(code=1008)  # anti-CSWSH (same as the chat)
        return
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:  # noqa: BLE001,S112: non-JSON frame: ignore it and continue  # nosec B112
                continue
            if not isinstance(msg, dict):
                continue
            task = (msg.get("task") or "").strip()
            project = (msg.get("project") or "").strip()
            if not task:
                continue
            # FIX for the "undefined" bug: HONESTY GUARD: without a VALID, EXISTING project nothing runs
            # (the worker writes files: it must never create phantom folders like ~/.../undefined/). The
            # view always creates the project first via POST /projects, so requiring existence is safe.
            # N15 improvement: ANY real subfolder of ~/agent_projects counts (including manually created
            # ones: spaces/uppercase/accents), worker.project_dir resolves it EXACTLY and confined.
            if not project or project == "undefined" or not worker.project_dir(cfg, project).is_dir():
                await websocket.send_json({"type": "worker_final", "state": "error",
                                           "summary": "Choose or create a project first (I didn't get a valid project)."})
                continue

            queue: asyncio.Queue = asyncio.Queue()
            END = {"type": "__end__"}

            def emitter(ev):
                loop.call_soon_threadsafe(queue.put_nowait, ev)

            def work():
                # The worker runs in a thread (produces events); the async handler drains them to the
                # WebSocket. Phase 2: uses the global MODEL and WORKDIR -> same care as the chat: ONE agent
                # at a time (_agent_lock, never two models) and capture/restore WORKDIR (don't leave it stuck
                # to the project).
                try:
                    with _agent_lock:
                        wd_orig = _agent.WORKDIR
                        try:
                            worker.run(task, project, emitter, cfg, _get_router())
                        finally:
                            _set_workdir(wd_orig)
                except Exception as e:  # noqa: BLE001: clean error to the view (never crashes the WS)
                    emitter({"type": "worker_final", "state": "error", "summary": f"{type(e).__name__}: {e}"})
                finally:
                    emitter(END)

            fut = loop.run_in_executor(None, work)
            send_ok = True
            while True:
                ev = await queue.get()
                if ev.get("type") == "__end__":
                    break
                if send_ok:
                    try:
                        await websocket.send_json(ev)
                    except Exception:  # noqa: BLE001: client disconnected: drain but don't send
                        send_ok = False
            await fut
            if not send_ok:
                break
    except WebSocketDisconnect:
        pass


def main():
    host = cfg.get("WEB_HOST") or "127.0.0.1"
    port = int(cfg.get("WEB_PORT") or "8080")
    print(f"\n  Agent visual chat:  http://{host}:{port}\n  (Ctrl-C to quit)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
