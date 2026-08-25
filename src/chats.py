"""Chats inside projects.

Data model:
- A PROJECT is a workspace: its documents, RAG index (`.knowledge/`) and project memory
  (`.memory.md`) are SHARED by all of its chats.
- A CHAT is a conversation (`conversation.jsonl`) with its own chat memory (the incremental
  summary `.conversation_summary.md` plus the last N turns), its lessons (`.lessons.md`) and
  its snapshots (time machine), all under `chats/<chat-id>/`.

Pure STORAGE module (never calls the model). Reuses `projects._slug` (path-safety) and its trash
pattern. Top priority: zero data loss. Migration takes a full BACKUP, is IDEMPOTENT and
TRANSACTIONAL (moves via atomic rename; if anything fails, the backup lets you roll back).
"""
import shutil
import time
from pathlib import Path

import projects

CHATS_DIR = "chats"                          # chat container inside the project
CHAT_ACTIVE_NAME = ".active_chat"            # per-project pointer to the active chat
CHATS_TRASH = ".trash_chats"             # deleted chats (recoverable), inside the project
BACKUP_DIR = ".backup_level14"               # migration backups, in the BASE dir
MAX_CHATS_TRASH = 10
MAX_BACKUPS = 10                             # migration backups to keep (bounded, never grows forever)
MAIN_CHAT = "principal"                      # chat that inherits the legacy conversation on migration

# Files/folders that in the OLD model lived at the project level and now belong to the CHAT.
# (Documents, `.knowledge/` and `.memory.md` are NOT here: they stay at project level.)
_LEGACY_FILES = ("conversation.jsonl", ".conversation_summary.md",
                 ".conversation_summary.md.tmp", ".lessons.md")
_LEGACY_DIRS = (".snapshots",)


def _slug(name):
    """SAFE chat id (path-safety: no traversal, no hidden names, no separators) and CASE-INSENSITIVE.
    Reuses the project sanitizer and additionally lowercases: on a case-insensitive filesystem
    (macOS/APFS by default), 'Ideas' and 'ideas' point to the SAME physical folder; normalizing to
    lowercase prevents two "different" chats from colliding (mixing their conversations) and, above
    all, prevents migration to the fixed 'principal' chat from clobbering a pre-existing 'Principal'
    (data loss)."""
    return projects._slug(name).lower()


# --- Paths ---------------------------------------------------------------------
def chats_dir(proj):
    return Path(proj) / CHATS_DIR


def chat_path(proj, chat_id):
    return chats_dir(proj) / _slug(chat_id)


def list_chats(proj):
    """Ids (slug) of existing chats, sorted. Ignores hidden entries (trash, pointers)."""
    d = chats_dir(proj)
    if not d.is_dir():
        return []
    return sorted(x.name for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))


def active_chat(proj):
    """Id of the project's active chat (persistent pointer), or the first existing one, or None."""
    proj = Path(proj)
    chats = list_chats(proj)
    f = proj / CHAT_ACTIVE_NAME
    if f.exists():
        try:
            n = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            n = ""
        if n and n in chats:
            return n
    return chats[0] if chats else None


def set_active_chat(proj, chat_id):
    (Path(proj) / CHAT_ACTIVE_NAME).write_text(_slug(chat_id), encoding="utf-8")


# --- CRUD ----------------------------------------------------------------------
def _auto_name(proj):
    """First free 'chat-N' (default name for a new chat)."""
    existing = set(list_chats(proj))
    i = 1
    while f"chat-{i}" in existing:
        i += 1
    return f"chat-{i}"


def create_chat(proj, name=None):
    """Create (idempotently) a chat and make it ACTIVE. Returns its slug. No name -> first free 'chat-N'."""
    proj = Path(proj)
    proj.mkdir(parents=True, exist_ok=True)
    slug = _slug(name) if (name or "").strip() else _auto_name(proj)
    chat_path(proj, slug).mkdir(parents=True, exist_ok=True)
    set_active_chat(proj, slug)
    return slug


def _prune_chats_trash(proj):
    trash = Path(proj) / CHATS_TRASH
    if not trash.is_dir():
        return
    entries = sorted((d for d in trash.iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime, reverse=True)
    for d in entries[MAX_CHATS_TRASH:]:
        shutil.rmtree(d, ignore_errors=True)


def delete_chat(proj, chat_id):
    """Delete a chat by MOVING it to the project trash (recoverable). Path-safe: the real destination
    must be a direct child of `chats/` (never outside, never hidden, never a symlink). If it was the
    active one, another chat is made active. Returns {ok, chat, active} or {ok:False, error}."""
    proj = Path(proj)
    slug = _slug(chat_id)
    if slug.startswith("."):
        return {"ok": False, "error": "invalid chat id"}
    path = chat_path(proj, slug)
    if not path.is_dir():
        return {"ok": False, "error": "the chat does not exist"}
    if path.is_symlink() or chats_dir(proj).is_symlink():   # neither the chat nor its container may be symlinks
        return {"ok": False, "error": "the chat is not a normal folder"}
    try:
        # Reinforced path-safety: the REAL destination must be a direct child of chats/ AND live INSIDE
        # the real project (if chats/ were a symlink to an external dir, both would resolve outside -> cut off).
        if (path.resolve().parent != chats_dir(proj).resolve()
                or Path(proj).resolve() not in path.resolve().parents):
            return {"ok": False, "error": "the chat is outside the chats folder"}
    except OSError:
        return {"ok": False, "error": "invalid path"}
    was_active = (active_chat(proj) == slug)
    trash = proj / CHATS_TRASH
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{slug}-{int(time.time() * 1000)}"
    i = 1
    while dest.exists():
        dest = trash / f"{slug}-{int(time.time() * 1000)}-{i}"
        i += 1
    try:
        shutil.move(str(path), str(dest))
    except OSError as e:
        return {"ok": False, "error": f"could not delete the chat ({type(e).__name__})"}
    _prune_chats_trash(proj)
    if was_active:
        remaining = list_chats(proj)
        if remaining:
            set_active_chat(proj, remaining[0])
        else:
            (proj / CHAT_ACTIVE_NAME).unlink(missing_ok=True)
    return {"ok": True, "chat": slug, "active": active_chat(proj)}


# --- Migration (CRITICAL: zero data loss) --------------------------------------
def _has_legacy(proj):
    """Does the project have conversation/chat-memory/lessons/snapshots at the PROJECT LEVEL
    (old, pre-chats model)?"""
    proj = Path(proj)
    return (any((proj / f).exists() for f in _LEGACY_FILES)
            or any((proj / d).is_dir() for d in _LEGACY_DIRS))


def is_migrated(proj):
    """Migrated = has at least one chat AND no legacy files left loose at the project level."""
    return bool(list_chats(proj)) and not _has_legacy(proj)


def _backup(proj, base):
    """Full backup of the project into BASE/.backup_level14/<slug>-<ts>/ (recoverable).
    Excludes only the backup/trash folders so they are not duplicated or recursed into."""
    proj = Path(proj)
    bdir = Path(base) / BACKUP_DIR
    bdir.mkdir(parents=True, exist_ok=True)
    dest = bdir / f"{proj.name}-{int(time.time() * 1000)}"
    i = 1
    while dest.exists():
        dest = bdir / f"{proj.name}-{int(time.time() * 1000)}-{i}"
        i += 1
    shutil.copytree(proj, dest, symlinks=True,
                    ignore=shutil.ignore_patterns(BACKUP_DIR, CHATS_TRASH))
    _prune_backups(bdir)                      # keep only the MAX_BACKUPS most recent (bounded)
    return dest


def _prune_backups(bdir):
    if not Path(bdir).is_dir():
        return
    entries = sorted((d for d in Path(bdir).iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime, reverse=True)
    for d in entries[MAX_BACKUPS:]:
        shutil.rmtree(d, ignore_errors=True)


def migrate_project(proj, base=None):
    """Migrate a project from the OLD model (conversation/chat-memory/lessons/snapshots at the project
    level) to the project->chats model: create `chats/principal/` and MOVE those files there. Documents,
    `.knowledge/` and `.memory.md` (PROJECT memory) stay at the project level.

    ZERO DATA LOSS:
    - Full BACKUP of the project BEFORE touching anything (reversible from the copy).
    - Transactional: the move uses `shutil.move` (ATOMIC rename on the same filesystem); if something
      fails midway, the backup lets you restore and the project itself stays as it was except for the
      already-moved items (which are intact inside `chats/principal/`).
    - Idempotent: running it twice does not duplicate or corrupt; a previous PARTIAL migration is
      COMPLETED (if an item is already in `principal`, the orphan legacy copy is removed).

    Returns {migrated, reason|chat|backup|items}."""
    proj = Path(proj)
    if not proj.is_dir():
        return {"migrated": False, "reason": "the project does not exist"}
    if base is None:
        base = proj.parent
    has_legacy = _has_legacy(proj)
    chats_ok = bool(list_chats(proj))
    if chats_ok and not has_legacy:
        return {"migrated": False, "reason": "already migrated"}
    if not has_legacy and not chats_ok:
        # New/empty project with no legacy data -> nothing to migrate (the chat is created on use).
        return {"migrated": False, "reason": "no legacy data"}
    # There IS legacy data to migrate (or a partial migration to complete). 1) Full BACKUP first.
    backup = _backup(proj, base)
    main = chat_path(proj, MAIN_CHAT)
    main.mkdir(parents=True, exist_ok=True)
    moved = []
    trash = proj / CHATS_TRASH
    for name in _LEGACY_FILES + _LEGACY_DIRS:
        src = proj / name
        if not src.exists():
            continue
        dst = main / name
        if dst.exists():
            # The destination ALREADY has this item (previous partial migration, or a legacy file that
            # REAPPEARED, which on a case-insensitive FS could collide by case with 'principal'/'Principal').
            # src is NEVER destroyed: it is moved to the project TRASH (recoverable) instead of deleted ->
            # ZERO LOSS even if its content differs from the migrated one (on top of the prior full backup).
            trash.mkdir(parents=True, exist_ok=True)
            dest = trash / f"legacy-{int(time.time() * 1000)}-{name.lstrip('.')}"
            i = 1
            while dest.exists():
                dest = trash / f"legacy-{int(time.time() * 1000)}-{i}-{name.lstrip('.')}"
                i += 1
            shutil.move(str(src), str(dest))
        else:
            shutil.move(str(src), str(dst))   # atomic rename (same filesystem)
        moved.append(name)
    _prune_chats_trash(proj)
    set_active_chat(proj, MAIN_CHAT)
    return {"migrated": True, "chat": MAIN_CHAT, "backup": str(backup), "items": moved}


def ensure_chat(proj, base=None):
    """Ensure the project is MIGRATED and has an ACTIVE chat; return its slug. Lazy: called when
    opening/using a project. Migrates if needed (with backup); if it is a new project with no chats,
    creates 'principal'. Idempotent and lossless."""
    proj = Path(proj)
    migrate_project(proj, base=base)             # no-op if already migrated or no legacy
    act = active_chat(proj)
    if act is None:
        act = create_chat(proj, MAIN_CHAT)       # new project with no chats -> create 'principal'
    return act
