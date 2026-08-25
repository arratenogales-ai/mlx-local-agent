"""Jupyter notebooks (.ipynb) handled DETERMINISTICALLY, with no external dependencies.

An `.ipynb` is JSON (nbformat v4). We parse and write the JSON DIRECTLY: `nbformat` is not
installed and is NOT needed (this avoids a download). The VALIDITY of the structure is guaranteed
by this script; the CONTENT of the cells comes from the model. Standing rules:

- Light context: when reading, we return the INDEX plus the source of each cell, CLIPPED, and huge
  outputs (images, long tables) are SUMMARIZED, never dumped into the context.
- Robust: a corrupt `.ipynb` returns an honest error, never blows up.
- Byte-safe on edit: the CONTENT of the other cells is preserved (parse, modify only the target
  cell, rewrite), with an atomic write. Nothing in the notebook is executed (out of scope).
- Conversion .ipynb <-> .py: "percent" format (`# %%`), deterministic and without jupytext.
"""
import json
import os
from pathlib import Path

MAX_SOURCE = 4000          # clip a cell's source on READ (light context)
MAX_OUTPUT = 400           # clip an output's text when summarizing it
MAX_CELLS = 500            # cap on listed cells


def _load(path: Path):
    """Return (nb_dict, None) or (None, error_message). NEVER raises on malformed JSON."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return None, f"could not read the file: {e}"
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"the .ipynb is not valid JSON (line {e.lineno}, col {e.colno}): {e.msg}"
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return None, "the .ipynb does not have the expected structure (missing 'cells')"
    return nb, None


def _source_str(source):
    """`source` in nbformat can be a string or a list of lines, normalize to a string."""
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def _summarize_outputs(outputs):
    """LIGHT summary of a code cell's outputs: types plus clipped text, without dumping
    images/base64 or huge tables."""
    if not isinstance(outputs, list) or not outputs:
        return ""
    parts = []
    for o in outputs:
        if not isinstance(o, dict):
            continue
        kind = o.get("output_type", "?")
        if kind == "stream":
            txt = _source_str(o.get("text", ""))
            parts.append(f"[stream {o.get('name', '')}] {txt[:MAX_OUTPUT].strip()}")
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {}) if isinstance(o.get("data"), dict) else {}
            if "text/plain" in data:
                parts.append(f"[result] {_source_str(data['text/plain'])[:MAX_OUTPUT].strip()}")
            imgs = [k for k in data if k.startswith("image/")]
            if imgs:
                parts.append(f"[image omitted: {', '.join(imgs)}]")
        elif kind == "error":
            parts.append(f"[error] {o.get('ename', '')}: {_source_str(o.get('evalue', ''))[:MAX_OUTPUT]}")
    return " | ".join(p for p in parts if p)


def read(path):
    """Notebook structure: {ok, n_cells, cells:[{i,type,lines,source,outputs}], language}.
    If the JSON is corrupt, {ok:False, error:...} (never crashes). The source is CLIPPED and the
    outputs SUMMARIZED (light context)."""
    nb, err = _load(Path(path))
    if err:
        return {"ok": False, "error": err}
    cells = nb.get("cells", [])
    language = (((nb.get("metadata") or {}).get("language_info") or {}).get("name")
                or ((nb.get("metadata") or {}).get("kernelspec") or {}).get("language") or "python")
    out_cells = []
    for i, c in enumerate(cells[:MAX_CELLS]):
        if not isinstance(c, dict):
            continue
        kind = c.get("cell_type", "?")
        source = _source_str(c.get("source"))
        n_lines = source.count("\n") + (1 if source and not source.endswith("\n") else 0)
        entry = {"i": i, "type": kind, "lines": n_lines,
                 "source": source[:MAX_SOURCE] + ("..." if len(source) > MAX_SOURCE else "")}
        if kind == "code":
            summary = _summarize_outputs(c.get("outputs"))
            if summary:
                entry["outputs"] = summary[:MAX_SOURCE]
        out_cells.append(entry)
    return {"ok": True, "n_cells": len(cells), "language": language,
            "truncated": len(cells) > MAX_CELLS, "cells": out_cells}


def format_result(structure):
    """COMPACT summary of the structure for the model/CLI (light context)."""
    if not structure.get("ok"):
        return f"ERROR reading the notebook: {structure.get('error')}"
    ls = [f"Notebook: {structure['n_cells']} cell(s), language {structure['language']}"
          + ("  (showing the first ones)" if structure.get("truncated") else "")]
    for c in structure["cells"]:
        head = f"[{c['i']}] {c['type']} | {c['lines']} line(s)"
        if c.get("outputs"):
            head += f" | outputs: {c['outputs'][:120]}"
        ls.append(head)
        body = c["source"].strip()
        if body:
            ls.append("    " + body.replace("\n", "\n    "))
    return "\n".join(ls)


_TEXT_TYPES = {"text", "markdown", "md"}
_CODE_TYPES = {"code", "python", "py"}


def _nb_cell(kind, source):
    """A valid nbformat v4 cell. `source` as a list of lines (canonical form)."""
    source = source if isinstance(source, str) else str(source or "")
    lines = source.splitlines(keepends=True) or [""]
    t = (kind or "code").strip().lower()
    if t in _TEXT_TYPES:
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def _nb_v4(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def _write_atomic(path: Path, nb: dict):
    tmp = path.with_name(path.name + ".__nb__.tmp")
    tmp.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def create(path, cells):
    """Write a VALID v4 `.ipynb` from `cells` = list of {type, source} (type: code|text).
    Returns a message. JSON validity is guaranteed by the script."""
    path = Path(path)
    if not isinstance(cells, list) or not cells:
        return "ERROR: 'cells' must be a non-empty list of {type, source}."
    nb_cells = []
    for c in cells:
        if not isinstance(c, dict):
            return "ERROR: each cell must be an object {type, source}."
        nb_cells.append(_nb_cell(c.get("type", "code"), c.get("source", "")))
    _write_atomic(path, _nb_v4(nb_cells))
    return f"OK: notebook created with {len(nb_cells)} cell(s) -> {path.name}"


def edit(path, index, action="replace", source=None, kind=None):
    """Edit ONE cell while preserving the CONTENT of the rest (parse, modify only that cell, rewrite,
    atomic). `action`: replace | insert | delete. Byte-safe: the other cells stay identical."""
    path = Path(path)
    nb, err = _load(path)
    if err:
        return f"ERROR: {err}"
    cells = nb["cells"]
    action = (action or "replace").strip().lower()
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return "ERROR: 'index' must be a cell number (0-indexed)."
    if action == "insert":
        if idx < 0 or idx > len(cells):
            return f"ERROR: index out of range for insert (0..{len(cells)})."
        new_cell = _nb_cell(kind or "code", source or "")
        # IDEMPOTENT: if an IDENTICAL cell already exists (same type and same source, INCLUDING the
        # empty one), it is not re-inserted. This way the correction loop (which re-runs the step)
        # does NOT duplicate already-added cells.
        new_source = _source_str(new_cell["source"]).strip()
        if any(isinstance(c, dict) and c.get("cell_type") == new_cell["cell_type"]
               and _source_str(c.get("source")).strip() == new_source for c in cells):
            return f"OK: the cell already exists; NOT duplicated (the notebook has {len(cells)} cell(s))."
        cells.insert(idx, new_cell)
    elif action == "delete":
        if not (0 <= idx < len(cells)):
            return f"ERROR: index out of range (0..{len(cells) - 1})."
        cells.pop(idx)
    elif action == "replace":
        if not (0 <= idx < len(cells)):
            return f"ERROR: index out of range (0..{len(cells) - 1})."
        prev = cells[idx] if isinstance(cells[idx], dict) else {}
        final_kind = kind or prev.get("cell_type", "code")
        cells[idx] = _nb_cell(final_kind, source if source is not None else _source_str(prev.get("source")))
    else:
        return "ERROR: invalid action (use replace | insert | delete)."
    _write_atomic(path, nb)
    return f"OK: cell {idx} {action} | the notebook now has {len(cells)} cell(s)"


def to_py(path_ipynb):
    """Notebook -> Python text in "percent" format (`# %%` code, `# %% [markdown]` text).
    Deterministic, without jupytext. Returns (text, None) or (None, error)."""
    nb, err = _load(Path(path_ipynb))
    if err:
        return None, err
    chunks = []
    for c in nb.get("cells", []):
        if not isinstance(c, dict):
            continue
        source = _source_str(c.get("source"))
        if c.get("cell_type") == "markdown":
            body = "\n".join("# " + ln for ln in source.splitlines()) or "#"
            chunks.append("# %% [markdown]\n" + body)
        else:
            chunks.append("# %%\n" + source.rstrip("\n"))
    return "\n\n".join(chunks) + "\n", None


def _cell_pairs(text):
    """[(type, source)] of a notebook given as JSON text, or None if it does not parse."""
    try:
        nb = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(nb, dict):
        return None
    return [(c.get("cell_type"), _source_str(c.get("source")))
            for c in nb.get("cells", []) if isinstance(c, dict)]


def render_cells_diff(before_text, after_text):
    """For VERIFICATION: deterministically renders which cells the task ADDED to a notebook (comparing
    by content before/after) with their FULL SOURCE. This way the verifier sees the real content (not a
    summary or the raw JSON) and knows which cells are the DELTA (it does not flag pre-existing ones).
    Returns '' if not applicable."""
    after = _cell_pairs(after_text)
    if after is None:
        return ""
    before = _cell_pairs(before_text) or []
    from collections import Counter
    available = Counter(before)
    added = []
    for pair in after:
        if available.get(pair, 0) > 0:
            available[pair] -= 1
        else:
            added.append(pair)
    ls = [f"The notebook has {len(after)} cell(s); this task ADDED {len(added)} "
          f"(the remaining {len(after) - len(added)} already existed, they are NOT a failure)."]
    for kind, source in added:
        ls.append(f"-- cell '{kind}' ADDED --\n{source.strip()}")
    return "\n".join(ls)


def to_markdown_doc(path_ipynb):
    """Notebook -> FULL Markdown for a DELIVERABLE (markdown as prose; code in ``` blocks).
    Unlike `read`, this does NOT truncate: it goes to an output file (not the model's context).
    Returns (markdown, None) or (None, error)."""
    nb, err = _load(Path(path_ipynb))
    if err:
        return None, err
    chunks = []
    for c in nb.get("cells", []):
        if not isinstance(c, dict):
            continue
        source = _source_str(c.get("source"))
        if c.get("cell_type") == "markdown":
            chunks.append(source)
        elif source.strip():
            chunks.append("```python\n" + source.rstrip("\n") + "\n```")
    return "\n\n".join(chunks) + "\n", None


def from_py(text):
    """Python text in "percent" format -> list of cells [{type, source}]. If there are no `# %%`
    markers, everything goes into ONE code cell. Deterministic."""
    lines = (text or "").splitlines()
    cells, buffer, kind = [], [], "code"

    def _flush():
        if buffer:
            body = "\n".join(buffer).strip("\n")
            if kind == "markdown":
                body = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                                 for ln in body.splitlines())
            cells.append({"type": kind, "source": body})

    has_markers = any(ln.strip().startswith("# %%") for ln in lines)
    if not has_markers:
        return [{"type": "code", "source": (text or "").rstrip("\n")}] if (text or "").strip() else []
    for ln in lines:
        if ln.strip().startswith("# %%"):
            _flush()
            buffer = []
            kind = "markdown" if "[markdown]" in ln.lower() else "code"
        else:
            buffer.append(ln)
    _flush()
    return cells


def convert(source_path, dest_path):
    """Convert `.ipynb`->`.py` or `.py`->`.ipynb` (by extension). Deterministic (without jupytext).
    Returns a message."""
    source_path, dest_path = Path(source_path), Path(dest_path)
    ext_src, ext_dst = source_path.suffix.lower(), dest_path.suffix.lower()
    if ext_src == ".ipynb" and ext_dst == ".py":
        text, err = to_py(source_path)
        if err:
            return f"ERROR: {err}"
        tmp = dest_path.with_name(dest_path.name + ".__conv__.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest_path)
        return f"OK: {source_path.name} -> {dest_path.name} (percent format, deterministic)"
    if ext_src == ".py" and ext_dst == ".ipynb":
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"ERROR: could not read {source_path.name}: {e}"
        cells = from_py(text)
        if not cells:
            return "ERROR: the .py is empty; there is nothing to convert."
        return create(dest_path, cells)
    return "ERROR: unsupported conversion (use .ipynb<->.py)."
