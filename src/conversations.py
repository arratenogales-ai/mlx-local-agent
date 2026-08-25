"""Level 10D: conversation search and history (convenience, deterministic).

Operates on a project's `conversation.jsonl` (the FULL history, one message per line):
- search(proj, query): the TURNS matching the query words (case/accent-insensitive),
  bounded. Light context: does NOT dump the whole history, only the trimmed relevant turns.
- export(proj, dest): dumps the conversation to `.md` (directly) or `.pdf/.docx/.html` (using
  the 4D pandoc). Deterministic. Degrades honestly if pandoc/LaTeX are missing.

100% local. Does not use the model (it is exact).
"""
import unicodedata
from pathlib import Path

import projects

MAX_TURNS = 30            # cap on turns returned per search (light context)
MAX_FRAG = 400            # trim each message when displaying/searching
_FORMATS = {".pdf", ".docx", ".html", ".odt", ".rtf", ".epub", ".tex"}


def _norm(s):
    """Lowercase without accents (for tolerant matching)."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def turns(proj):
    """Groups the history into TURNS (user message -> agent reply). Tolerates an odd
    history (a lone message is paired with '')."""
    messages = projects.load_conversation(Path(proj))
    out, pending = [], None
    for m in messages:
        if m.get("role") == "user":
            if pending is not None:
                out.append({"user": pending, "reply": ""})
            pending = m.get("content", "")
        elif m.get("role") == "assistant":
            out.append({"user": pending or "", "reply": m.get("content", "")})
            pending = None
    if pending is not None:
        out.append({"user": pending, "reply": ""})
    return out


def search(proj, query, limit=MAX_TURNS):
    """Turns whose text (user+reply) contains ALL the words in `query`. Returns
    [{n, user, reply}] with the texts trimmed (light context)."""
    words = [_norm(p) for p in (query or "").split() if p.strip()]
    if not words:
        return []
    found = []
    for i, t in enumerate(turns(proj), 1):
        haystack = _norm(t["user"] + " \n " + t["reply"])
        if all(p in haystack for p in words):
            found.append({"n": i,
                          "user": t["user"][:MAX_FRAG],
                          "reply": t["reply"][:MAX_FRAG]})
            if len(found) >= limit:
                break
    return found


def format_search(results, query):
    if not results:
        return f"No turns containing \"{query}\" in the conversation."
    ls = [f"{len(results)} turn(s) with \"{query}\":"]
    for r in results:
        ls.append(f"\n[turn {r['n']}]")
        ls.append("  You: " + r["user"].replace("\n", " ")[:200])
        ls.append("  Agent: " + r["reply"].replace("\n", " ")[:200])
    return "\n".join(ls)


def _to_markdown(proj):
    name = Path(proj).name
    ls = [f"# Conversation: {name}\n"]
    for i, t in enumerate(turns(proj), 1):
        ls.append(f"## Turn {i}\n")
        ls.append(f"**You:** {t['user']}\n")
        ls.append(f"**Agent:** {t['reply']}\n")
    return "\n".join(ls)


def export(proj, dest):
    """Exports the project's conversation to `dest` (.md directly; .pdf/.docx/... via pandoc).
    Returns a message. Honest: if pandoc/LaTeX are missing, it warns and leaves no half-written files."""
    proj, dest = Path(proj), Path(dest)
    turns_ = turns(proj)
    if not turns_:
        return "ERROR: there is no conversation to export in this project."
    md = _to_markdown(proj)
    ext = dest.suffix.lower()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if ext in ("", ".md", ".markdown", ".txt"):
        dest.write_text(md, encoding="utf-8")
        return f"OK: conversation ({len(turns_)} turns) exported to {dest.name}"
    if ext not in _FORMATS:
        return f"ERROR: format '{ext}' not supported (use .md, .pdf, .docx, .html...)."
    import deliverables                              # reuses the SAFE pandoc (temp mkstemp + cleans up dest on failure)
    err = deliverables._pandoc_from_md(md, dest)
    if err:
        return err
    return f"OK: conversation ({len(turns_)} turns) exported to {dest.name}"
