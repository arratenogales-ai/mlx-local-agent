"""Generate polished deliverables (.docx/.pdf/.html) from notes or a notebook.

Reuses the 4D pandoc path (local and free) with a clean template: a standalone document
(`--standalone`), the title as metadata and, optionally, a table of contents (`--toc`). The source
is whatever the user points at (light context: no content is invented or over-dumped). Deterministic;
the model writes the notes, this turns them into a presentable document. Honest: if pandoc/LaTeX are
missing, it warns and does not leave half-written files.
"""
import os
import subprocess  # nosec B404 - for `pandoc` (notes -> .docx/.pdf): fixed list, no shell
import tempfile
from pathlib import Path

import notebooks

PANDOC_TIMEOUT = 90
_FORMATS = {".pdf", ".docx", ".html", ".odt", ".rtf", ".epub", ".tex"}
_TEXT = {"", ".md", ".markdown", ".txt"}


def _md_from_source(path: Path):
    """FULL Markdown from the source: .ipynb -> prose+code; .md/.txt/.rst -> as-is."""
    ext = path.suffix.lower()
    if ext == ".ipynb":
        return notebooks.to_markdown_doc(path)
    if ext in (".md", ".markdown", ".txt", ".rst", ".text"):
        try:
            return path.read_text(encoding="utf-8", errors="replace"), None
        except OSError as e:
            return None, f"could not read {path.name}: {e}"
    return None, f"unsupported source: '{ext}' (use .ipynb, .md or .txt)"


def generate(source, target, title=None, with_toc=False):
    """Generate a deliverable at `target` from `source` (.ipynb/.md/.txt). `.md` is written
    directly; `.pdf/.docx/.html/...` are converted with pandoc (clean template: standalone + title +
    optional toc). Returns a message."""
    source, target = Path(source), Path(target)
    if not source.is_file():
        return f"ERROR: the source {source.name} does not exist."
    md, err = _md_from_source(source)
    if err:
        return f"ERROR: {err}"
    if not (md or "").strip():
        return "ERROR: the source is empty; nothing to generate."
    ext = target.suffix.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if ext in _TEXT:
        header = f"# {title}\n\n" if title else ""
        target.write_text(header + md, encoding="utf-8")
        return f"OK: Markdown deliverable written to {target.name}"
    if ext not in _FORMATS:
        return f"ERROR: format '{ext}' not supported (use .pdf, .docx, .html, .md...)."
    extra = (["--toc"] if with_toc else []) + (["--metadata", f"title={title}"] if title else [])
    err = _pandoc_from_md(md, target, extra)
    if err:
        return err
    return f"OK: deliverable '{title or target.stem}' generated at {target.name}"


def _pandoc_from_md(md, target: Path, extra=None):
    """Convert `md` to `target` with pandoc (--standalone + `extra`) SAFELY and ATOMICALLY:
    pandoc writes to an output TEMP file (mkstemp, O_EXCL) and only on success is it moved into place
    with os.replace (it NEVER writes through a symlink at `target`, nor leaves a half-written file).
    Cleans up all temp files. Returns None on success, or an honest ERROR message on failure."""
    fd_in, tmp_in = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}_in_", suffix=".md")
    fd_out, tmp_out = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}_out_", suffix=target.suffix)
    os.close(fd_out)                               # pandoc will write here; we only needed to reserve the name
    try:
        with os.fdopen(fd_in, "w", encoding="utf-8") as fh:
            fh.write(md)
        cmd = ["pandoc", tmp_in, "-o", tmp_out, "--standalone"] + list(extra or [])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PANDOC_TIMEOUT)  # noqa: S603 - `cmd` is a fixed list (pandoc + our own tempfile paths), no shell  # nosec B603
        except FileNotFoundError:
            return "ERROR: pandoc is not installed; generate a .md or install pandoc (local and free)."
        except subprocess.TimeoutExpired:
            return "ERROR: pandoc took too long; try a .md or .docx."
        if proc.returncode != 0:
            return (f"ERROR: pandoc could not generate {target.suffix} ({(proc.stderr or '').strip()[:200]}). "
                    f"A .pdf needs a LaTeX engine; if it is missing, generate .docx or .html.")
        os.replace(tmp_out, target)                # atomic; replaces the target without following a symlink
        tmp_out = None                             # already moved: do not delete it in the finally
        return None
    finally:
        for t in (tmp_in, tmp_out):
            if t:
                try:
                    os.unlink(t)
                except OSError:
                    pass
