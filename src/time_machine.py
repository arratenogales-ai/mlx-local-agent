"""Level 8A time machine: snapshots plus undo.

Before a task that MODIFIES files, a snapshot of the working directory is taken; the user can UNDO
the last task and leave the files EXACTLY as they were, byte for byte.

Safety (the most important part: an undo must never lose user data):
- Undo acts ONLY on the REAL DIFF of the task (created/modified/deleted by IT, recorded right after
  the task), never "everything not in the snapshot", so a file the user creates afterwards is NOT
  deleted.
- Fail-closed: if the manifest or the diff cannot be read (corrupt/absent), the undo ABORTS (deletes
  nothing). The JSON files are written atomically (.tmp + os.replace).
- Scoped to `base` (never outside; excludes agent metadata, .git, .snapshots, caches). Handles
  file<->directory transitions, preserves permissions, and treats symlinks as links (does not follow
  them).

Uniform copy-based design (works the same with and without git; does not touch the user's repo beyond
the files). Files larger than 5MB are NOT copied (they are tracked, but flagged as 'not_restorable').
"""

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

SNAP_DIR = ".snapshots"
MAX_SNAPSHOTS = 12
MAX_SNAP_FILE = 5_000_000

_EXCLUDE = {
    SNAP_DIR,
    ".knowledge",
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    ".DS_Store",
    ".memory.md",
    ".lessons.md",
    ".conversation_summary.md",
    ".conversation_summary.md.tmp",
    "conversation.jsonl",
    ".active_project",
    # Level 14: chats and their metadata live under the project but are NOT "documents", so they are
    # never snapshotted (a project snapshot must not copy the conversations/memories of every chat).
    "chats",
    ".active_chat",
    ".trash_chats",
    ".backup_level14",
}


def _is_excluded(rel: Path) -> bool:
    return any(part in _EXCLUDE for part in rel.parts)


def _project_files(base: Path):
    """RELATIVE paths of the 'project' files under `base` (includes symlinks as entries), pruning
    excluded dirs and never leaving `base`."""
    base = Path(base)
    base_real = base.resolve()
    for root, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [d for d in sorted(dirs) if d not in _EXCLUDE]
        for name in sorted(files):
            if name in _EXCLUDE:
                continue
            p = Path(root) / name
            rel = p.relative_to(base)
            if _is_excluded(rel):
                continue
            if p.is_symlink():
                yield rel  # the link lives inside base: tracked as a link
                continue
            try:
                if base_real not in p.resolve().parents:  # (should not happen) outside the project
                    continue
            except OSError:
                continue
            yield rel


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _signature_of(base: Path, rel: Path):
    """(hash, is_symlink, target|None) of file `rel`: content hash, or the target hash if it is a link."""
    p = base / rel
    if p.is_symlink():
        target = os.readlink(p)
        return _hash_bytes(("L:" + target).encode("utf-8")), True, target
    return _hash_file(p), False, None


def _snaps_dir(base: Path) -> Path:
    d = Path(base) / SNAP_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(dest: Path, obj):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dest)  # atomic: never leaves a half-written JSON


def _read_json(p: Path):
    """Returns the object, or None if it could NOT be read/parsed (fail-closed: distinct from empty {})."""
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _next_id(base: Path) -> str:
    ids = [int(d.name) for d in _snaps_dir(base).iterdir() if d.is_dir() and d.name.isdigit()]
    return f"{(max(ids) + 1 if ids else 1):04d}"


def _current_signature(base: Path):
    """{relpath: hash} of the CURRENT state (to compare against a snapshot). Symlinks by their target."""
    out = {}
    for rel in _project_files(Path(base)):
        try:
            out[str(rel)] = _signature_of(Path(base), rel)[0]
        except OSError:
            continue
    return out


def _current_dirs(base: Path):
    """RELATIVE paths of the project directories (including EMPTY ones), so we know which existed
    BEFORE the task and do NOT delete them on undo (even if they end up empty)."""
    base = Path(base)
    out = set()
    for root, dirs, _ in os.walk(base, followlinks=False):
        dirs[:] = [d for d in dirs if d not in _EXCLUDE]
        for d in dirs:
            rel = (Path(root) / d).relative_to(base)
            if not _is_excluded(rel):
                out.add(str(rel))
    return out


def create_snapshot(base, task="", label="", store=None) -> str:
    """Snapshots the CURRENT state of the project files under `base` (the DOCUMENTS). Saves the
    snapshot in `store` (default `base`): Level 14 passes the CHAT directory as `store` so snapshots
    live per chat, even though the snapshotted documents belong to the PROJECT. Returns the id."""
    base = Path(base)
    store = Path(store) if store is not None else base
    sid = _next_id(store)
    dest = _snaps_dir(store) / sid
    (dest / "files").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for rel in _project_files(base):
        src = base / rel
        try:
            h, is_link, target = _signature_of(base, rel)
        except (OSError, ValueError):
            # Level 20 (D3c): it could not even be read, so it is RECORDED anyway (without a copy). If it
            # were skipped, after the task it would count as "created" and the undo would DELETE a user file.
            manifest[str(rel)] = {"type": "file", "hash": None, "copied": False}
            continue
        if is_link:
            manifest[str(rel)] = {"type": "symlink", "hash": h, "target": target, "copied": True}
            continue
        try:
            st = src.stat()
            entry = {
                "type": "file",
                "hash": h,
                "copied": st.st_size <= MAX_SNAP_FILE,
                "size": st.st_size,
                "mode": stat.S_IMODE(st.st_mode),
            }
        except OSError:
            entry = {"type": "file", "hash": h, "copied": False}
        if entry["copied"]:
            try:
                d = dest / "files" / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, d)  # exact bytes
            except OSError:  # e.g. disk full: tracked, but NOT restorable
                entry["copied"] = False
        manifest[str(rel)] = entry
    _write_json(dest / "manifest.json", manifest)
    _write_json(
        dest / "meta.json",
        {
            "id": sid,
            "ts": time.time(),
            "task": (task or "")[:300],
            "label": label,
            "n_files": len(manifest),
            "dirs": sorted(_current_dirs(base)),
        },
    )  # prior dirs (incl. empty)
    _prune(store)
    return sid


def _read_manifest(base: Path, sid: str):
    return _read_json(_snaps_dir(base) / sid / "manifest.json")  # dict or None


def _read_changes(base: Path, sid: str):
    return _read_json(_snaps_dir(base) / sid / "changes.json")  # dict or None


def read_meta(base, sid):
    return _read_json(_snaps_dir(Path(base)) / sid / "meta.json") or {}


def record_changes(base, sid, store=None) -> bool:
    """AFTER the task: computes the DIFF the TASK made (created/modified/deleted) by comparing the
    CURRENT state with the manifest (pre-task) and saves it. Returns True if the task changed anything.
    This is what scopes the undo to what the task did (never to the user's later work)."""
    base = Path(base)
    store = Path(store) if store is not None else base
    man = _read_manifest(store, sid)
    if man is None:
        return False
    current = _current_signature(base)
    created = sorted(set(current) - set(man))
    # Level 20 (review): an UNREADABLE file (recorded with hash None) that is still on disk is NOT "deleted":
    # before, every turn saved a phantom snapshot that pushed the real undo history out. (A file that became a
    # folder IS a deletion.)
    deleted = sorted(k for k in set(man) - set(current) if not ((base / k).is_file() or (base / k).is_symlink()))
    modified = sorted(k for k in man if k in current and current[k] != man[k]["hash"])
    after = {k: current[k] for k in created + modified}  # Level 20 (D3d): the state RIGHT AFTER the task
    _write_json(
        _snaps_dir(store) / sid / "changes.json",
        {"created": created, "modified": modified, "deleted": deleted, "after": after},
    )
    return bool(created or deleted or modified)


def list_snapshots(base):
    """Existing snapshots, newest to oldest. Tolerates a corrupt meta.json (uses the directory's own
    id). Orders by NUMERIC id (not by string)."""
    base = Path(base)
    out = []
    for d in _snaps_dir(base).iterdir():
        if d.is_dir() and d.name.isdigit():
            m = read_meta(base, d.name) or {}
            m.setdefault("id", d.name)
            out.append(m)
    return sorted(out, key=lambda m: int(m.get("id") or 0), reverse=True)


def last_snapshot(base):
    ss = list_snapshots(base)
    return ss[0]["id"] if ss else None


def plan_undo(base, sid, store=None):
    """What the undo would change, WITHOUT applying it (for confirmation). Uses the task's DIFF, not
    'everything not in the snapshot'. Returns None (fail-closed) if the manifest or the diff cannot be
    read. Keys: `revert` (modified), `recreate` (deleted), `delete` (created by the task and still
    present), `not_restorable` (huge files not copied), `changed_after` (Level 20: files the task
    created/modified that the user changed AGAIN afterwards, so the undo leaves them alone)."""
    base = Path(base)
    store = Path(store) if store is not None else base
    man = _read_manifest(store, sid)
    changes = _read_changes(store, sid)
    if man is None or changes is None:
        return None
    current = _current_signature(base)

    def _copied(r):
        return bool(man.get(r, {}).get("copied"))

    # Level 20 (D3d): if a file the task created/modified changed AGAIN AFTERWARDS (the user edited it),
    # the undo does NOT touch it: it goes to `changed_after` so the user decides. (Old snapshots without
    # "after" -> previous behavior.)
    after = changes.get("after")

    def _untouched(r):
        return after is None or r not in after or current.get(r) == after[r]

    mods, created = changes.get("modified", []), changes.get("created", [])
    revert = [r for r in mods if r in current and _copied(r) and _untouched(r)]
    recreate = [r for r in changes.get("deleted", []) if r not in current and _copied(r)]
    delete = [r for r in created if r in current and _untouched(r)]
    changed_after = [r for r in mods + created if r in current and not _untouched(r)]
    not_rest = [r for r in mods if r in current and not _copied(r) and _untouched(r)] + [
        r for r in changes.get("deleted", []) if r not in current and not _copied(r)
    ]
    return {
        "revert": sorted(revert),
        "recreate": sorted(recreate),
        "delete": sorted(delete),
        "not_restorable": sorted(not_rest),
        "changed_after": sorted(changed_after),
    }


def _within(base: Path, p: Path) -> bool:
    try:
        pr = p.resolve()
        br = base.resolve()
        return pr == br or br in pr.parents
    except OSError:
        return False


def _clear_dest(dest: Path):
    """Frees the spot to rewrite `dest` as a file/link (handles the type transition)."""
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest, ignore_errors=True)


def _restore_one(files: Path, rel: str, entry: dict, dest: Path):
    """Restores ONE file/link ATOMICALLY and FAIL-CLOSED: builds the good version in a TEMP file
    (verifying the source FIRST) and only then puts it in place with os.replace. If the snapshot copy
    is missing or truncated, it ABORTS that file WITHOUT having touched the live version, so no data is
    lost (audit-2 bug)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".mt_tmp")
    _clear_dest(tmp)  # in case a tmp from a previous attempt was left
    try:
        if entry.get("type") == "symlink":
            target = entry.get("target")
            if not target:  # no saved target, do not destroy the live one
                raise FileNotFoundError(f"symlink without target: {rel}")
            os.symlink(target, tmp)
        else:
            source = files / rel
            if not source.is_file():  # snapshot copy absent/corrupt, ABORT
                raise FileNotFoundError(f"snapshot copy missing: {rel}")
            shutil.copyfile(source, tmp)  # byte for byte, into the temp file first
            expected = entry.get("hash")
            if expected and _hash_file(tmp) != expected:  # Level 20 (D3b): damaged/truncated copy -> ABORT
                raise OSError(f"the snapshot copy of {rel} is damaged (hash mismatch)")
            try:
                if "mode" in entry:
                    os.chmod(tmp, entry["mode"])
            except OSError:
                pass
        # ATOMIC swap: os.replace overwrites a live file/symlink with no gap; if the dest is a
        # DIRECTORY (dir->file transition), we remove it just before.
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():  # if something failed before the replace, leave no junk
            try:
                _clear_dest(tmp)
            except OSError:
                pass


def _delete_one(dest: Path):
    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest, ignore_errors=True)  # the task created a dir here (transition)
    elif dest.is_file():
        dest.unlink()


def undo(base, sid=None, store=None):
    """Leaves the files EXACTLY as in snapshot `sid` (or the latest): reverts the ones the task
    modified, recreates the ones it deleted, and deletes the ones it created (byte for byte). SAFE:
    fail-closed if the snapshot data is not trustworthy; only acts INSIDE `base` and on the task's DIFF.
    `store` (default `base`) is where the snapshot lives (Level 14: the chat directory). Returns the
    applied plan, or None if it could not be done."""
    base = Path(base)
    store = Path(store) if store is not None else base
    if sid is None:
        sid = last_snapshot(store)
    if not sid or not (_snaps_dir(store) / sid).is_dir():
        return None
    plan = plan_undo(base, sid, store=store)
    if plan is None:
        return None
    man = _read_manifest(store, sid)
    files = _snaps_dir(store) / sid / "files"
    failed = []
    # Level 20 (D3a): ORDER. First DELETE what the task created (and its new empty folders), and THEN
    # restore. Before it was the other way round: if the task replaced the folder `a/` (with a/x.txt) by a
    # file `a`, restoring a/x.txt failed (mkdir clashed with the file `a`, error swallowed) and then `a`
    # was deleted, so BOTH were lost.
    # 1) delete ONLY the files the task created (and still as the task left them), never other files
    for rel in plan["delete"]:
        dest = base / rel
        if not _within(base, dest) or _is_excluded(Path(rel)):
            continue
        try:
            _delete_one(dest)
        except OSError:
            failed.append(rel)
    # 2) clean up ONLY the folders the TASK created: ancestors of the deleted files that did NOT exist
    # before the snapshot (per the recorded dirs). Never a folder that pre-existed for the user (even
    # if it ends up empty), fixes the deletion of unrelated folders (audit-2).
    prior_dirs = set(read_meta(store, sid).get("dirs", []))
    candidates = set()
    for rel in plan["delete"]:
        p = Path(rel).parent
        while str(p) not in (".", ""):
            candidates.add(str(p))
            p = p.parent
    for rd in sorted(candidates, key=lambda s: s.count(os.sep), reverse=True):  # deepest first
        if rd in prior_dirs or _is_excluded(Path(rd)):  # user's pre-existing dir, do not touch
            continue
        d = base / rd
        if not _within(base, d):
            continue
        try:
            if d.is_dir() and not d.is_symlink() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            continue
    # 3) revert/recreate (byte for byte, verified by hash) the files copied by the snapshot
    for rel in plan["revert"] + plan["recreate"]:
        dest = base / rel
        if not _within(base, dest) or _is_excluded(Path(rel)):
            continue
        try:
            # an intermediate folder that is now a FILE created by the task was already deleted in (1)
            _restore_one(files, rel, man.get(rel, {}), dest)
        except OSError:
            failed.append(rel)
    plan["failed"] = sorted(failed)  # Level 20 (D3e): honest, what could NOT be undone
    return plan


def delete_snapshot(base, sid):
    d = _snaps_dir(Path(base)) / str(sid)
    if d.is_dir() and str(sid).isdigit():
        shutil.rmtree(d, ignore_errors=True)


def _prune(base):
    for m in list_snapshots(base)[MAX_SNAPSHOTS:]:
        delete_snapshot(base, m["id"])
