"""Level 9A: SKILLS infrastructure: pluggable capabilities with lightweight context.

A skill is a folder `skills/<name>/SKILL.md` with a header (`name`, `when_to_use`, optional
`script`) and a body of instructions plus examples that the model READS AND FOLLOWS when the
task fits. Optionally, a deterministic script does the critical part.

Principles:
- Lightweight context: a cheap CATALOG (only name + `when_to_use` for each skill) is always
  available, but the full INSTRUCTIONS of a skill are loaded and injected only in the turn that
  needs them, and dropped afterward. They are never all loaded at once.
- Deterministic over the 14B's judgment: SELECTION of which skill applies is done via TF-IDF
  (reusing the 8B tokenizer, `knowledge._tokenize`) over `when_to_use`, with a THRESHOLD: if
  nothing beats it, NO skill fires (zero false triggers).
- Safety: skills are data, not a free pass. Names are path-safe (no traversal); a malformed
  header is SKIPPED (nothing breaks); the declared `script` runs through the normal guard and
  tools (here it is only validated and located; the orchestrator runs it).
- Toggle: `SKILLS=false` in config.env disables the whole subsystem.
"""
import math
import os
import re
from collections import Counter
from pathlib import Path

from knowledge import _tokenize   # 8B: same tokenizer, for reuse and consistency

ROOT = Path(__file__).resolve().parent.parent          # code lives in src/; skills in the repo root
DEFAULT_DIR = "skills"
SKILL_FILE = "SKILL.md"
DEFAULT_THRESHOLD = 0.30          # minimum TF-IDF cosine to fire (zero false triggers; wide margin)
DEFAULT_MAX_CHARS = 4000          # cap on injected instructions (lightweight context)

# English stop words: they carry no signal for choosing a skill and add noise (false triggers).
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "with",
    "that", "this", "these", "those", "it", "its", "my", "mine", "your", "yours", "his",
    "her", "hers", "our", "ours", "their", "me", "us", "them", "is", "are", "be", "was",
    "were", "been", "more", "less", "very", "already", "if", "yes", "no", "not", "how",
    "what", "which", "do", "does", "did", "make", "made", "give", "put", "want", "can",
    "could", "please", "get", "let",
    # generic objects: on their own they must not trigger a skill (e.g. "with the TEXT: ...")
    "text", "paragraph",
}

# Safe skill name: no traversal, no path separators or hidden entries.
_RX_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Skill:
    """LIGHTWEIGHT catalog entry: name + when_to_use (+ script), and where it lives. The full
    instructions are NOT loaded here (lightweight context): read them on demand with instructions()."""
    __slots__ = ("name", "when_to_use", "script", "dir")

    def __init__(self, name, when_to_use, script, dir):
        self.name = name
        self.when_to_use = when_to_use
        self.script = script
        self.dir = dir


# --- Config -------------------------------------------------------------------
def enabled(cfg):
    """Is the subsystem active? SKILLS toggle (on by default)."""
    return str((cfg or {}).get("SKILLS", "true")).strip().lower() == "true"


def _skills_dir(cfg):
    d = str((cfg or {}).get("SKILLS_DIR", "") or DEFAULT_DIR).strip()
    p = Path(os.path.expanduser(d))
    return p if p.is_absolute() else (ROOT / p)


def _threshold(cfg):
    try:
        return float((cfg or {}).get("SKILL_THRESHOLD", "") or DEFAULT_THRESHOLD)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


def _max_chars(cfg):
    try:
        return max(200, int((cfg or {}).get("SKILL_MAX_CHARS", "") or DEFAULT_MAX_CHARS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS


# --- Robust SKILL.md parsing --------------------------------------------------
def _parse(text):
    """Split the header (--- key: value ---) from the body. Returns (meta, body) or None if the
    header is malformed or missing the essentials (name/when_to_use). Robust: never raises."""
    if not text:
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return None
    meta = {}
    for line in lines[1:close]:
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    if not meta.get("name") or not meta.get("when_to_use"):
        return None
    body = "\n".join(lines[close + 1:]).strip()
    return meta, body


def _read_md(skill_dir):
    try:
        return (skill_dir / SKILL_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --- Catalog (lightweight) ----------------------------------------------------
def _scan(base: Path):
    """Valid skills from ONE directory (one per folder with SKILL.md). Robust: malformed or with an
    unsafe name is SKIPPED (with a warning), never breaks anything."""
    skills = []
    if not base.is_dir():
        return skills
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        try:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not _RX_NAME.match(entry.name):
                _warn(f"unsafe skill name, skipping: {entry.name!r}")
                continue
            parsed = _parse(_read_md(entry))
            if not parsed:
                _warn(f"malformed skill, skipping: {entry.name}")
                continue
            meta, _body = parsed
            name = meta.get("name", "").strip()   # the declared name must also be safe
            if not _RX_NAME.match(name or ""):
                _warn(f"unsafe declared name, skipping: {name!r}")
                continue
            script = (meta.get("script") or "").strip() or None
            if script and (os.sep in script or "/" in script or ".." in script):
                _warn(f"script with unsafe path in {entry.name}, ignoring the script")
                script = None
            skills.append(Skill(name, meta["when_to_use"], script, entry))
        except OSError as e:                      # odd I/O on one folder: skip it, keep going
            _warn(f"error reading skill {entry.name}: {e}")
            continue
    return skills


def _project_skills_dir(project_dir):
    """The `skills/` folder INSIDE a project (or None if absent or missing). Level 9D."""
    if not project_dir:
        return None
    p = Path(project_dir) / DEFAULT_DIR
    return p if p.is_dir() else None


def load_catalog(cfg, project_dir=None):
    """LIGHTWEIGHT catalog: the GLOBAL skills + (9D) those of the active project, if it has its own
    `skills/` folder. Project skills are ADDED and, on a name collision, WIN (customization). They
    never mix across projects: only the folder of the passed `project_dir` is included."""
    base = _skills_dir(cfg)
    catalog_by_name = {}
    for s in _scan(base):                         # globals first
        catalog_by_name[s.name] = s
    proj = _project_skills_dir(project_dir)
    if proj is not None and proj.resolve() != base.resolve():
        for s in _scan(proj):                     # project skills are added and override on collision
            catalog_by_name[s.name] = s
    return sorted(catalog_by_name.values(), key=lambda s: s.name)


def _warn(msg):
    import sys
    print(f"skills warning: {msg}", file=sys.stderr)


# --- Deterministic selection (TF-IDF + threshold) -----------------------------
def _tokens(text):
    """8B tokens, minus the stop words (they sharpen the signal, zero false triggers)."""
    return [t for t in _tokenize(text or "") if t not in _STOP]


def _phrases(when_to_use):
    """`when_to_use` is a LIST of trigger phrases (separated by commas or newlines). Each phrase is a
    candidate mini-query."""
    return [f.strip() for f in re.split(r"[,\n;]+", when_to_use or "") if f.strip()]


def _score(query, docs):
    """TF-IDF cosine (same math as the 8B) between `query` and each `when_to_use`, but at the
    TRIGGER-PHRASE level: a skill's score is the MAXIMUM cosine over its phrases. IDF is computed over
    the corpus of ALL phrases, so common words ('text', 'that') lose weight and distinctive ones
    ('humanize', 'AI', 'robotic') gain it, giving a sharp separation and zero false triggers. Returns
    a list of scores aligned with `docs`."""
    q_tok = _tokens(query)
    if not q_tok or not docs:
        return [0.0] * len(docs)
    phrases_per_doc = [[_tokens(f) for f in _phrases(d)] for d in docs]
    corpus = [ft for phrases in phrases_per_doc for ft in phrases if ft]
    if not corpus:
        return [0.0] * len(docs)
    df = Counter()
    for ft in corpus:
        df.update(set(ft))
    n = max(1, len(corpus))
    idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def _vec(toks):
        return {t: f * idf.get(t, 0.0) for t, f in Counter(toks).items()}

    q_vec = _vec(q_tok)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    scores = []
    for phrases in phrases_per_doc:
        best = 0.0
        for ft in phrases:
            if not ft:
                continue
            d_vec = _vec(ft)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_vec.get(t, 0.0) * w for t, w in d_vec.items())
            best = max(best, dot / (q_norm * d_norm))
        scores.append(best)
    return scores


def select(message, cfg, catalog=None, project_dir=None):
    """Pick the relevant skill for `message` (one per turn) DETERMINISTICALLY over the global catalog
    + the project's (9D). Returns the Skill if it beats the threshold; if none fits, None (zero false
    triggers). Respects the toggle."""
    if not enabled(cfg):
        return None
    cat = catalog if catalog is not None else load_catalog(cfg, project_dir)
    if not cat:
        return None
    scores = _score(message, [s.when_to_use for s in cat])
    best_i = max(range(len(cat)), key=lambda i: scores[i])
    return cat[best_i] if scores[best_i] >= _threshold(cfg) else None


def by_name(name, cfg, catalog=None, project_dir=None):
    """Skill with that EXACT name from the catalog (global + project, 9D), or None. To FORCE a skill
    from the web (9C): skips the threshold (the user picked it by hand), but respects toggle and
    catalog."""
    if not name or not enabled(cfg):
        return None
    cat = catalog if catalog is not None else load_catalog(cfg, project_dir)
    return next((s for s in cat if s.name == name), None)


def _description(when_to_use):
    """Short label for the web panel: the first trigger phrase (without inflating the catalog)."""
    phrases = _phrases(when_to_use)
    return phrases[0][:80] if phrases else ""


def web_catalog(cfg, project_dir=None):
    """LIGHTWEIGHT catalog for the web panel: [{name, description}]. Includes the active project's
    skills (9D) if its folder is passed. Respects the toggle."""
    if not enabled(cfg):
        return []
    return [{"name": s.name, "description": _description(s.when_to_use)}
            for s in load_catalog(cfg, project_dir)]


# --- On-demand injection ------------------------------------------------------
def instructions(skill, cfg):
    """Load the BODY of `skill`'s SKILL.md (instructions + examples), capped at SKILL_MAX_CHARS
    (lightweight context). Called only in the turn that uses the skill. '' on failure."""
    parsed = _parse(_read_md(skill.dir))
    if not parsed:
        return ""
    _meta, body = parsed
    cap = _max_chars(cfg)
    return body[:cap] if len(body) > cap else body


def script_path(skill):
    """ABSOLUTE path of the script declared by the skill, validated to stay inside its folder
    (path-safety). None if it declares no script, it does not exist, or it would escape the folder."""
    if not skill.script:
        return None
    p = (skill.dir / skill.script).resolve()
    try:
        base = skill.dir.resolve()
        if base not in p.parents or not p.is_file():
            return None
    except OSError:
        return None
    return p
