"""Generate polished deliverables (.docx/.pdf/.html) from notes or a notebook.

Reuses the 4D pandoc path (local and free) with a clean template: a standalone document
(`--standalone`), the title as metadata and, optionally, a table of contents (`--toc`). The source
is whatever the user points at (light context: no content is invented or over-dumped). Deterministic;
the model writes the notes, this turns them into a presentable document. Honest: if pandoc/LaTeX are
missing, it warns and does not leave half-written files.
"""

import html.parser
import json
import os
import re
import subprocess  # nosec B404 - for `pandoc` (notes -> .docx/.pdf): fixed list, no shell
import tempfile
from pathlib import Path

from . import notebooks

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
    err = _pandoc_from_md(md, target, extra, root=source.parent)
    if err:
        return err
    return f"OK: deliverable '{title or target.stem}' generated at {target.name}"


# Level 20 (S3): pandoc EMBEDS into the .docx/.pdf/.html the images the Markdown references. A
# `![x](/secret/path)` (or `![x](~/.ssh/id_rsa)`, or a URL that would make pandoc go out to the internet) put
# files from OUTSIDE the project into the deliverable -- and with WEB_SEARCH the text may come from a hostile
# page. `--sandbox` would stop it, but it also breaks the project's LEGITIMATE images and does not stop LaTeX's
# \input{}; so: (1) image references are sanitized (only relative paths INSIDE the project), and (2) the
# dangerous raw TeX/HTML constructs are removed (see below).
# Level 22: TWO layers. (1) The regexes below clean the TEXT: they leave a readable notice and remove the obvious
# dangerous constructs. They are LINEAR on purpose (CodeQL: the old ones were quadratic, up to 120 s on a crafted
# text): no class can run past the next '[', '<', '\' or line end. (2) What DECIDES is pandoc's OWN tree
# (_sanitize_tree): images outside the project, raw TeX outside a short allowlist and raw HTML that runs or loads
# something are replaced there, whatever their spelling (a blocklist cannot catch '\csname input\endcsname', which
# read a file from outside the project into a PDF).
_RX_IMG_MD = re.compile(r'!\[([^\[\]]*)\]\(\s*<?([^)\s<>\[]+)>?(?:\s+(?:"[^"\[\]\n]*"|\'[^\'\[\]\n]*\'))?\s*\)')
_RX_IMG_REF = re.compile(r"!\[([^\[\]]*)\](?:\[([^\[\]]*)\])?(?!\()")
_RX_REF_DEF = re.compile(r"^[ ]{0,3}\[([^\]\n]+)\]:[ \t]*<?(\S+?)>?(?=\s|$)", re.M)
_INPUT_FORMAT = "markdown"
# Level 20 (review): raw TeX/HTML is KEPT (tables with <br>, \newpage, <!-- notes -->, H<sub>2</sub>O use it;
# disabling it wholesale printed them as literal text) and only the DANGEROUS constructs are removed.
_RX_TEX_DANGEROUS = re.compile(
    r"\\(?:input|include|includegraphics|verbatiminput|lstinputlisting|openin|read|"
    r"write18|immediate|catcode|usepackage|includepdf|csname)\b(?:\[[^\[\]\\]*\])?(?:\{[^{}\\]*\})?"
)
_RX_HTML_DANGEROUS = re.compile(
    r"<\s*(script|style|iframe|object|embed|link|meta|base|frame|frameset)\b[^<>]*>(?:[^<]*<\s*/\s*\1\s*>)?",
    re.IGNORECASE,
)
_RX_IMG_HTML = re.compile(r"<\s*img\b[^<>]*?\bsrc\s*=\s*[\"']?([^\"'\s<>]++)[^<>]*+>", re.IGNORECASE)
# Raw TeX that may stay: only these harmless commands, with plain arguments (no '\', '^' or other active characters).
_RX_TEX_SAFE = re.compile(
    r"(?:\s*(?:\\(?:newpage|pagebreak|clearpage|noindent|linebreak|newline|centering|raggedright|raggedleft"
    r"|smallskip|medskip|bigskip|vfill|hfill|today|LaTeX|TeX|par)(?![A-Za-z])"
    r"|\\[vh]space\*?\{-?[0-9.]+\s*(?:cm|mm|in|pt|em|ex)\}"
    r"|\\(?:textbf|textit|emph|underline|textsc|texttt)\{[^{}\\$%#&~^_]*\}"
    r"|\\\\))*\s*"
)
_HTML_BAD_TAGS = {"script", "style", "iframe", "object", "embed", "link", "meta", "base", "frame", "frameset", "form"}
_HTML_MEDIA = {"img", "source", "video", "audio", "track", "input", "image"}


def _image_allowed(ref, root):
    if ref.lower().startswith("data:image/"):  # embedded in the text itself: reads no files and no network
        return True
    if re.match(r"^[a-z][a-z0-9+.-]*:", ref, re.I) or ref.startswith(("~", "//")):
        return False  # http(s)/file/...: no network and no absolute paths
    try:
        real = (Path(root) / ref).resolve()
        return Path(root).resolve() in real.parents and real.is_file()
    except (OSError, ValueError):
        return False


def sanitize_images(md, root):
    """Keeps ONLY the images that are files INSIDE `root` (the project). The rest are replaced by a
    visible notice in the document (honest: they do not silently vanish). Level 20 (review): it also strips
    the dangerous raw TeX/HTML (file reads, shell escapes, embedded/loaded HTML, <img> from outside)."""
    md = md or ""

    def _inline(m):
        return m.group(0) if _image_allowed(m.group(2), root) else f"[image omitted: {m.group(1) or m.group(2)}]"

    md = _RX_IMG_MD.sub(_inline, md)
    used = {(m.group(2) or m.group(1)).strip().lower() for m in _RX_IMG_REF.finditer(md)}

    def _definition(m):
        if m.group(1).strip().lower() in used and not _image_allowed(m.group(2), root):
            return f"[{m.group(1)}]: #image-omitted"
        return m.group(0)

    md = _RX_REF_DEF.sub(_definition, md)
    md = _RX_TEX_DANGEROUS.sub("", md)
    md = _RX_HTML_DANGEROUS.sub("", md)
    return _RX_IMG_HTML.sub(lambda m: m.group(0) if _image_allowed(m.group(1), root) else "[image omitted]", md)


class _RawHTMLCheck(html.parser.HTMLParser):
    """Level 22: is a chunk of raw HTML dangerous? (a real parser, not a regex: entities and odd spacing do not
    fool it). Dangerous = a tag that runs or loads something, an on*= handler, a javascript:/vbscript:/data:text
    link, or media whose source is outside the project."""

    def __init__(self, root):
        super().__init__(convert_charrefs=True)
        self.root, self.bad = root, False

    def handle_starttag(self, tag, attrs):
        if tag in _HTML_BAD_TAGS:
            self.bad = True
        for name, value in attrs:
            name, value = (name or "").lower(), re.sub(r"[\s\x00-\x1f]", "", value or "")
            if name.startswith("on") or re.match(r"(?i)(javascript|vbscript|data:text)", value):
                self.bad = True
            if tag in _HTML_MEDIA and name in ("src", "srcset", "poster", "href", "xlink:href"):
                if not _image_allowed(value.split(",")[0], self.root):
                    self.bad = True

    handle_startendtag = handle_starttag


def _raw_html_is_dangerous(text, root):
    check = _RawHTMLCheck(root)
    try:
        check.feed(text)
        check.close()
    except Exception:  # noqa: BLE001 - if it cannot be parsed, it does not go in (fail closed)
        return True
    return check.bad


def _sanitize_tree(ast, root):
    """Level 22: the DECIDING layer, on pandoc's OWN tree (pandoc decides what is an image or raw code, so a
    creative spelling cannot slip past). Replaced by a visible notice: images whose source is not allowed
    (an alt text with nested brackets, '![a [b] c](/outside.png)', also by reference, got past the regexes and
    pandoc embedded an image from OUTSIDE the project); raw TeX outside _RX_TEX_SAFE; raw HTML that runs or loads
    something; and any other raw format (OpenXML, RTF...: nobody checks them). Iterative (no recursion limit)."""
    notice = {"Image": "[image omitted]", "tex": "[TeX omitted]", "html": "[HTML omitted]"}
    stack = [ast]
    while stack:
        node = stack.pop()
        items = node if isinstance(node, list) else list(node.values()) if isinstance(node, dict) else []
        for i, x in enumerate(items):
            text, kind = None, x.get("t") if isinstance(x, dict) else None
            if kind == "Image" and not _image_allowed(str(x["c"][2][0]), root):
                text = notice["Image"]
            elif kind in ("RawInline", "RawBlock"):
                fmt, raw = str(x["c"][0]).lower(), str(x["c"][1])
                if fmt in ("tex", "latex"):
                    text = None if _RX_TEX_SAFE.fullmatch(raw) else notice["tex"]
                elif fmt in ("html", "html4", "html5"):
                    text = notice["html"] if _raw_html_is_dangerous(raw, root) else None
                else:
                    text = f"[{fmt} omitted]"
            if text is not None:
                if not isinstance(node, list):  # fail CLOSED: never leave it in
                    raise ValueError("raw content or image in an unexpected place")
                inline = {"t": "Str", "c": text}
                node[i] = {"t": "Para", "c": [inline]} if kind == "RawBlock" else inline
            elif isinstance(x, (list, dict)):
                stack.append(x)
    return ast


def _run_pandoc(args, root):
    """One pandoc call (fixed list, no shell) inside `root`. Returns (CompletedProcess, None) or (None, ERROR)."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argument list (pandoc + our own tempfile paths), no shell
            ["pandoc", *args],  # noqa: S607 # pandoc from PATH, like everywhere else in the project
            capture_output=True,
            text=True,
            timeout=PANDOC_TIMEOUT,  # nosec B603
            cwd=str(root),
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None, "ERROR: pandoc is not installed; generate a .md or install pandoc (local and free)."
    except subprocess.TimeoutExpired:
        return None, "ERROR: pandoc took too long; try a .md or .docx."
    return proc, None


def _pandoc_from_md(md, target: Path, extra=None, root=None):
    """Convert `md` to `target` with pandoc (--standalone + `extra`) SAFELY and ATOMICALLY:
    pandoc writes to an output TEMP file (mkstemp, O_EXCL) and only on success is it moved into place
    with os.replace (it NEVER writes through a symlink at `target`, nor leaves a half-written file).
    Cleans up all temp files. Returns None on success, or an honest ERROR message on failure.
    Level 20 (S3): images are sanitized to `root` (the project; default: the target's folder) and the
    dangerous raw TeX/HTML is removed. It is the ONE pandoc path, used by both create_document and generate."""
    root = Path(root or target.parent).resolve()
    md = sanitize_images(md, root)  # Level 20 (S3)
    fd_in, tmp_in = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}_in_", suffix=".md")
    fd_out, tmp_out = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}_out_", suffix=target.suffix)
    os.close(fd_out)  # pandoc will write here; we only needed to reserve the name
    fd_ast, tmp_ast = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.stem}_ast_", suffix=".json")
    try:
        with os.fdopen(fd_in, "w", encoding="utf-8") as fh:
            fh.write(md)
        # Level 22: pandoc itself reads the text, the images are checked on ITS tree, and the document is
        # written from that checked tree (never from the raw text).
        proc, err = _run_pandoc(["-f", _INPUT_FORMAT, tmp_in, "-t", "json"], root)
        if err is None and proc.returncode == 0:
            try:
                ast = _sanitize_tree(json.loads(proc.stdout), root)
            except (ValueError, LookupError, TypeError) as e:
                return f"ERROR: could not check the document's images and raw code ({type(e).__name__}); nothing was generated."
            with os.fdopen(fd_ast, "w", encoding="utf-8") as fh:
                fd_ast = None
                json.dump(ast, fh)
            cmd = ["-f", "json", tmp_ast, "-o", tmp_out, "--standalone", f"--resource-path={root}"] + list(extra or [])
            proc, err = _run_pandoc(cmd, root)
        if err:
            return err
        if proc.returncode != 0:
            return (
                f"ERROR: pandoc could not generate {target.suffix} ({(proc.stderr or '').strip()[:200]}). "
                f"A .pdf needs a LaTeX engine; if it is missing, generate .docx or .html."
            )
        os.replace(tmp_out, target)  # atomic; replaces the target without following a symlink
        tmp_out = None  # already moved: do not delete it in the finally
        return None
    finally:
        if fd_ast is not None:
            os.close(fd_ast)
        for t in (tmp_in, tmp_out, tmp_ast):
            if t:
                try:
                    os.unlink(t)
                except OSError:
                    pass
