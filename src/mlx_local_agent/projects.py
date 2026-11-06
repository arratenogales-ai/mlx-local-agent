"""Named projects with persistent conversation (lightweight context).

A PROJECT is a folder (inside AGENT_PROJECTS_DIR) with its files, its `.memory.md`
(existing 4E memory) and its `conversation.jsonl` (FULL history, one message per line,
append-only). The full history is stored on disk (nothing is lost, it can be re-read), but
each turn the model only receives a compact SUMMARY plus the LAST N verbatim messages,
never the whole history. This is the 4E memory idea extended to the conversation: it keeps
the per-call context small and CONSTANT even when the conversation is long.

Pure STORAGE module: it never calls the model except in `update_summary` (which receives the client).
"""

import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

CONV_NAME = "conversation.jsonl"  # FULL history (append, one line per message)
SUMMARY_NAME = ".conversation_summary.md"  # compact summary (lightweight context), re-summarized
ACTIVE_NAME = ".active_project"  # pointer (in the base) to the active project
MAX_SUMMARY_CHARS = 1500  # summary cap (it is not a log)
MAX_INJECTED_MSG = 800  # cap per recent injected message (lightweight context)
_LAST_N_DEF = 6  # number of recent verbatim messages (default)

# Project name to SAFE folder (no path traversal, no hidden, no separators).
_RX_NAME = re.compile(r"[^A-Za-z0-9_.\- ]+")


def _slug(name):
    # Folds accents (Cafe->Cafe), replaces disallowed chars with '-', collapses dashes, and prevents
    # it from starting with '.' (hidden) or containing path separators (anti path-traversal).
    s = unicodedata.normalize("NFD", (name or "").strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = _RX_NAME.sub("-", s).replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-.")
    return s[:64] or "project"


def last_n(cfg):
    try:
        return max(1, min(40, int(cfg.get("CONV_LAST_N") or _LAST_N_DEF)))
    except (TypeError, ValueError):
        return _LAST_N_DEF


def base_dir(cfg):
    """BASE folder for projects (created if missing). Configurable in config.env."""
    d = (cfg.get("AGENT_PROJECTS_DIR") or "~/agent_projects").strip()
    p = Path(os.path.expanduser(d)).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(cfg, name):
    """REAL folder of project `name`. If the base has a folder with that EXACT name (including ones
    created BY HAND: spaces, uppercase, accents...), it is THAT one; otherwise the usual slug path.

    Level 20 (D2): everything used to go through the slug, but list_projects returns the REAL names, so
    with the folders "My Project" (by hand) and "My-Project", deleting "My Project" sent "My-Project" (the
    WRONG one) to the trash, and a hand-made folder could not be opened. This logic used to live only in
    worker.project_dir; now it is the whole system's.
    PATH-SAFETY: only the last component (kills '../' and '/x'), never hidden, and CONFINED to the base
    (a symlink pointing outside is ignored and falls back to the slug path)."""
    base = base_dir(cfg)
    clean = Path(str(name or "")).name.strip()
    if clean and not clean.startswith("."):
        d = base / clean
        try:
            if d.is_dir() and d.resolve().parent == base:
                return d
        except OSError:
            pass
    return base / _slug(name)


def exists(cfg, name):
    return project_path(cfg, name).is_dir()


def create_project(cfg, name):
    """Create (idempotently) the project and return its path."""
    p = project_path(cfg, name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_projects(cfg):
    """REAL names of the project folders, sorted (the app's are slugs; hand-made ones, as they are).
    Ignores hidden ones. project_path() resolves any of them to ITS exact folder."""
    b = base_dir(cfg)
    return sorted(d.name for d in b.iterdir() if d.is_dir() and not d.name.startswith("."))


def active_project(cfg):
    """Name of the active project (persistent pointer), or None if there is none / it does not exist."""
    f = base_dir(cfg) / ACTIVE_NAME
    if f.exists():
        try:
            n = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if n and (base_dir(cfg) / n).is_dir():
            return n
    return None


def set_active(cfg, name):
    """Set the active project (pointer in the base). Level 20 (D2): stores the name of the REAL folder
    (it used to store the slug, so a hand-made folder ended up pointing to one that does not exist)."""
    (base_dir(cfg) / ACTIVE_NAME).write_text(project_path(cfg, name).name, encoding="utf-8")


TRASH_NAME = ".trash"  # inside the base: deleted projects (recoverable)
MAX_TRASH = 10  # number of deleted projects the trash keeps


_RX_MS_STAMP = re.compile(r"-(\d{13})(?=-|$)")  # "<name>-<epoch ms>[-n]" (unique_destination) or "legacy-<ms>-<x>"


def unique_destination(folder, prefix):
    """NEW path "<folder>/<prefix>-<epoch ms>[-n]" that does not exist yet. The NAME's timestamp is the
    date the entry went into the trash/backup (prune_by_age uses it). Level 20: this loop used to be
    copied three times (project trash, chat trash and migration backups)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dest, i = folder / f"{prefix}-{int(time.time() * 1000)}", 1
    while dest.exists():
        dest = folder / f"{prefix}-{int(time.time() * 1000)}-{i}"
        i += 1
    return dest


def _age(d):
    """When `d` ENTERED the trash/backup: the name's stamp (reliable) or, if it has none, its mtime."""
    m = _RX_MS_STAMP.search(d.name)
    if m:
        return int(m.group(1)) / 1000
    try:
        return d.stat().st_mtime
    except OSError:
        return 0.0


def prune_by_age(folder, maximum, keep=None):
    """Keep only the `maximum` MOST RECENT entries of `folder` and NEVER delete `keep` (the one just
    created). Level 20 (D1): entries used to be sorted by the folder's st_mtime, which is NOT the deletion
    date (shutil.move keeps the project's mtime and copytree copies it): a project with no recent changes
    came out "the oldest" and was DELETED the very moment it was sent to the "recoverable" trash. Now the
    name's stamp (unique_destination) decides."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    keep = Path(keep).resolve() if keep else None
    entries = sorted((d for d in folder.iterdir() if d.is_dir() and not d.is_symlink()), key=_age, reverse=True)
    extra = [d for d in entries if keep is None or d.resolve() != keep][max(0, maximum - (1 if keep else 0)) :]
    for d in extra:
        shutil.rmtree(d, ignore_errors=True)


def _prune_trash(base, keep=None):
    """Keep only the MAX_TRASH most recent deletions so the trash does not grow unbounded."""
    prune_by_age(Path(base) / TRASH_NAME, MAX_TRASH, keep)


def delete_project(cfg, name):
    """Delete a project (the whole folder: files + .memory.md + conversation.jsonl + everything) by
    MOVING it to the TRASH (.trash/) inside the base, so it is recoverable and safer than an rm. If
    it was the ACTIVE project, leave another (or none) active consistently. Returns
    {ok, name, active} or {ok:False, error} (honest failure, no crash).

    SECURITY (this is destructive): the name is resolved with project_path (exact folder or slug,
    path-traversal-safe); it only acts INSIDE AGENT_PROJECTS_DIR, on a REAL DIRECTORY that is a direct
    child of the base (never outside, never hidden, never following symlinks)."""
    base = base_dir(cfg)  # already .resolve()
    proj = project_path(cfg, name)  # Level 20 (D2): the EXACT listed folder (not a re-slug)
    if proj.name.startswith("."):  # (should not happen: project_path never returns hidden) defense
        return {"ok": False, "error": "invalid project name"}
    if not proj.is_dir():
        return {"ok": False, "error": "the project does not exist"}
    if proj.is_symlink():  # never follow a symlink to delete
        return {"ok": False, "error": "the project is not a normal folder"}
    try:  # the real target must be a DIRECT child of the base
        if proj.resolve().parent != base:
            return {"ok": False, "error": "the project is outside the base folder"}
    except OSError:
        return {"ok": False, "error": "invalid path"}
    was_active = active_project(cfg) == proj.name  # BEFORE moving (so we only reassign if needed)
    # Move to the trash (recoverable). UNIQUE target (ms + counter) so we do not OVERWRITE a previous
    # entry if the same project is deleted twice in the same second.
    dest = unique_destination(base / TRASH_NAME, _slug(proj.name))
    try:
        shutil.move(str(proj), str(dest))
    except OSError as e:
        return {"ok": False, "error": f"could not delete the project ({type(e).__name__})"}
    _prune_trash(base, keep=dest)  # trash stays bounded, and what was JUST deleted is never pruned
    # If I deleted the ACTIVE one, leave another existing project (or none) active, no dangling refs.
    if was_active:
        remaining = list_projects(cfg)
        if remaining:
            set_active(cfg, remaining[0])
        else:
            (base / ACTIVE_NAME).unlink(missing_ok=True)
    return {"ok": True, "name": proj.name, "active": active_project(cfg)}


# --- Conversation (full history on disk) --------------------------------------
def load_conversation(proj):
    """Read the FULL history (list of {role, content}). Tolerant of corrupted lines."""
    f = Path(proj) / CONV_NAME
    if not f.exists():
        return []
    msgs = []
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m:
            msgs.append({"role": m["role"], "content": str(m["content"])})
    return msgs


def save_turn(proj, user_msg, response):
    """Append the turn (user message + agent response) to the history. The pair is written in a SINGLE
    call (not two writes), so a user message is never left without its response if the process dies in
    between."""
    f = Path(proj) / CONV_NAME
    pair = (
        json.dumps({"role": "user", "content": user_msg}, ensure_ascii=False)
        + "\n"
        + json.dumps({"role": "assistant", "content": response}, ensure_ascii=False)
        + "\n"
    )
    with f.open("a", encoding="utf-8") as fh:
        fh.write(pair)


# --- LIGHTWEIGHT context (what the model actually sees: summary + last N) ------
def read_summary(proj):
    f = Path(proj) / SUMMARY_NAME
    if not f.exists():
        return ""
    try:
        return f.read_text(encoding="utf-8", errors="replace").strip()[:MAX_SUMMARY_CHARS]
    except (OSError, UnicodeError):
        return ""


def _clip(text):
    text = text or ""
    return text if len(text) <= MAX_INJECTED_MSG else text[:MAX_INJECTED_MSG] + " ...[clipped]"


def conversation_context(proj, n):
    """LIGHTWEIGHT context messages to inject into the model: a `system` message with the SUMMARY (if
    any) plus the LAST N verbatim messages (each clipped). Never the whole history, so the size stays
    bounded and CONSTANT even as the conversation grows."""
    msgs = []
    summary = read_summary(proj)
    if summary:
        msgs.append(
            {
                "role": "system",
                "content": "SUMMARY of this project's conversation (context to resume; do NOT "
                "repeat it verbatim):\n" + summary,
            }
        )
    for m in load_conversation(proj)[-n:]:
        msgs.append({"role": m["role"], "content": _clip(m["content"])})
    return msgs


SUMMARY_PROMPT = (
    "You maintain the SUMMARY of a conversation (to resume it without re-reading the whole thing). "
    "Given the PREVIOUS summary and the LAST turn (user + agent), return the UPDATED summary, VERY "
    "BRIEF (max ~150 words): what it is about, decisions, state and pending items. MERGE and SUMMARIZE; "
    "drop what is obsolete; never accumulate the history. Reply ONLY with the summary."
)


def update_summary(client, model, proj, user_msg, response):
    """Re-summarize the conversation INCREMENTALLY (previous summary + last turn), like the 4E memory:
    the input for THIS call is bounded (it does not read the whole history), so the cost does not grow
    with the conversation. Best-effort: if it fails, it does not break the turn."""
    previous = read_summary(proj)
    user = (
        f"Previous summary:\n{previous or '(empty)'}\n\n"
        f"Last turn:\nUser: {_clip(user_msg)}\nAgent: {_clip(response)}\n\n"
        "Return the UPDATED, compact summary."
    )
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}],
        )
        new_summary = (r.choices[0].message.content or "").strip() if r.choices else ""
    except Exception:  # noqa: BLE001 the summary must never take down the turn
        return
    if not new_summary:
        return
    if len(new_summary) > MAX_SUMMARY_CHARS:
        new_summary = new_summary[:MAX_SUMMARY_CHARS].rstrip() + "\n...[clipped]"
    try:
        tmp = (Path(proj) / SUMMARY_NAME).with_suffix(".md.tmp")
        tmp.write_text(new_summary, encoding="utf-8")
        os.replace(tmp, Path(proj) / SUMMARY_NAME)
    except OSError:
        pass
