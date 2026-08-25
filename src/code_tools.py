"""Deterministic tools over Python code: a structure map (via `ast`) and a linter
(ruff if available; otherwise basic checks with `ast`). The script does the exact work; the model explains it.

- Map: functions, classes (with methods) and imports of a `.py`, with line numbers (via `ast`, exact).
  A `.py` with a syntax error returns an honest error with the line, without crashing.
- Linter: `ruff check` (fast, already installed) -> [{line, code, message}]; with a timeout (anti-hang).
  If `ruff` is missing or fails, it degrades honestly: `ast` detects at least the syntax error and a
  few safe problems (bare except, mutable default argument, comparison with `== None`).
"""
import ast
import difflib
import json
import os
import shutil
import subprocess  # nosec B404 - for the `ruff` linter (fixed list, no shell)
from pathlib import Path

RUFF_TIMEOUT = 20
_EXCLUDE = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", "node_modules", ".venv",
            "venv", ".snapshots", ".knowledge", ".trash"}
MAX_RESULTS = 200        # cap on matches/entries (light context)
MAX_FILES = 400
MAX_DIFF_LINES = 400


def _args_of(node):
    a = node.args
    names = [p.arg for p in (a.posonlyargs + a.args)]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


def code_map(code):
    """Structure of a Python module: {ok, functions, classes, imports} or {ok:False, error, line}
    on SyntaxError. `code` is the source text."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        return {"ok": False, "error": f"syntax error: {e.msg}", "line": e.lineno or 0}
    functions, classes, imports = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "line": node.lineno, "args": _args_of(node)})
        elif isinstance(node, ast.ClassDef):
            methods = [{"name": m.name, "line": m.lineno, "args": _args_of(m)}
                       for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({"name": node.name, "line": node.lineno, "methods": methods})
        elif isinstance(node, ast.Import):
            imports += [al.name for al in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports += [f"{mod}.{al.name}" for al in node.names]
    return {"ok": True, "functions": functions, "classes": classes, "imports": sorted(set(imports))}


def format_map(m):
    """Compact summary of the map (light context)."""
    if not m.get("ok"):
        return f"ERROR: {m.get('error')} (line {m.get('line')})"
    ls = []
    if m["imports"]:
        ls.append("imports: " + ", ".join(m["imports"][:40]))
    for f in m["functions"]:
        ls.append(f"def {f['name']}({', '.join(f['args'])})  . L{f['line']}")
    for c in m["classes"]:
        ls.append(f"class {c['name']}  . L{c['line']}")
        for me in c["methods"]:
            ls.append(f"    def {me['name']}({', '.join(me['args'])})  . L{me['line']}")
    return "\n".join(ls) or "(no top-level functions, classes or imports)"


def _ruff_available():
    return shutil.which("ruff") is not None


def _linter_ruff(path):
    try:
        r = subprocess.run(["ruff", "check", "--output-format", "json", "--", str(path)],  # noqa: S603,S607 - fixed command (no shell); `--` separates the (controlled) path; ruff via PATH on purpose  # nosec B603 B607
                           capture_output=True, text=True, timeout=RUFF_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (r.stdout or "").strip()
    if not output:
        return []                                   # no warnings (ruff returns rc!=0 but empty stdout)
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    warnings = []
    for d in data:
        loc = d.get("location") or {}
        warnings.append({"line": loc.get("row", 0), "code": d.get("code") or "", "message": d.get("message", "")})
    return warnings


def _linter_ast(code):
    """Honest degradation without ruff: syntax plus a few SAFE problems via `ast`."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        return [{"line": e.lineno or 0, "code": "E999", "message": f"syntax error: {e.msg}"}]
    warnings = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            warnings.append({"line": n.lineno, "code": "E722", "message": "bare except (catches everything)"})
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.args.defaults + n.args.kw_defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    warnings.append({"line": d.lineno, "code": "B006",
                                     "message": "MUTABLE default argument (list/dict/set)"})
        if isinstance(n, ast.Compare):
            for op, comp in zip(n.ops, n.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) and comp.value is None:
                    warnings.append({"line": n.lineno, "code": "E711", "message": "compares with None: use 'is'/'is not'"})
    return warnings


def linter(path, code=None):
    """Linter warnings for `path`: [{line, code, message}]. Uses ruff if available; otherwise basic ast.
    Also returns the 'source' (ruff|ast) to be honest about what was used."""
    if _ruff_available():
        warnings = _linter_ruff(path)
        if warnings is not None:
            return {"source": "ruff", "warnings": warnings}
    if code is None:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                code = fh.read()
        except OSError as e:
            return {"source": "ast", "warnings": [{"line": 0, "code": "", "message": f"could not read: {e}"}]}
    return {"source": "ast", "warnings": _linter_ast(code)}


def format_linter(res):
    source, warnings = res.get("source", "?"), res.get("warnings", [])
    if not warnings:
        return f"no warnings ({source})"
    ls = [f"L{a['line']} . {a['code']} . {a['message']}" for a in sorted(warnings, key=lambda a: a["line"])]
    ls.append(f"\n{len(warnings)} warning(s) [{source}]"
              + ("" if source == "ruff" else " - without ruff: only basic checks (honest degradation)"))
    return "\n".join(ls)


def _py_files(base: Path):
    """Paths of the .py files under `base` (pruning system/environment dirs). Bounded."""
    base = Path(base)
    if base.is_file():
        return [base] if base.suffix == ".py" else []
    out = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in sorted(dirs) if d not in _EXCLUDE and not d.startswith(".")]
        for n in sorted(files):
            if n.endswith(".py"):
                out.append(Path(root) / n)
                if len(out) >= MAX_FILES:
                    return out
    return out


def find_symbol(name, base):
    """DEFINITIONS of a symbol (function/class/method) called `name` under `base`, via `ast`
    (exact). Returns [{file, line, type}]. If there are no definitions, falls back to a search for
    occurrences (grep) so the user is not left without leads. Light context: only file:line."""
    base = Path(base)
    target = (name or "").strip()
    if not target:
        return {"ok": False, "error": "provide the symbol name"}
    defs, occurrences = [], []
    for f in _py_files(base):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(base)) if base.is_dir() else f.name
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
                    defs.append({"file": rel, "line": node.lineno, "type": "function/method"})
                elif isinstance(node, ast.ClassDef) and node.name == target:
                    defs.append({"file": rel, "line": node.lineno, "type": "class"})
        if len(defs) < MAX_RESULTS:
            for i, ln in enumerate(src.splitlines(), 1):
                if target in ln:
                    occurrences.append({"file": rel, "line": i, "text": ln.strip()[:120]})
                    if len(occurrences) >= MAX_RESULTS:
                        break
    return {"ok": True, "symbol": target, "definitions": defs[:MAX_RESULTS],
            "occurrences": occurrences[:MAX_RESULTS]}


def format_search(res):
    if not res.get("ok"):
        return f"ERROR: {res.get('error')}"
    ls = []
    if res["definitions"]:
        ls.append(f"Definitions of '{res['symbol']}':")
        ls += [f"  {d['file']}:{d['line']}  ({d['type']})" for d in res["definitions"]]
    else:
        ls.append(f"No DEFINITION of '{res['symbol']}'. Occurrences:")
        ls += [f"  {a['file']}:{a['line']}: {a['text']}" for a in res["occurrences"][:40]]
        if not res["occurrences"]:
            ls.append("  (none)")
    return "\n".join(ls)


def project_structure(base):
    """Compact map of a project: for each .py, its top-level functions/classes (name + line).
    Light context: the structure, never the whole code. Useful to generate a README."""
    base = Path(base)
    files = _py_files(base)
    entries = []
    for f in files:
        try:
            m = code_map(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        rel = str(f.relative_to(base)) if base.is_dir() else f.name
        if not m["ok"]:
            entries.append({"file": rel, "error": m["error"], "line": m["line"]})
            continue
        entries.append({"file": rel,
                        "functions": [f"{x['name']} (L{x['line']})" for x in m["functions"]],
                        "classes": [f"{c['name']} (L{c['line']})" for c in m["classes"]]})
    return {"ok": True, "n_files": len(files), "files": entries}


def format_structure(res):
    ls = [f"Project: {res['n_files']} .py file(s)"]
    for e in res["files"]:
        if e.get("error"):
            ls.append(f"\n{e['file']}  - {e['error']} (L{e['line']})")
            continue
        ls.append(f"\n{e['file']}")
        if e["functions"]:
            ls.append("  functions: " + ", ".join(e["functions"][:30]))
        if e["classes"]:
            ls.append("  classes: " + ", ".join(e["classes"][:30]))
    return "\n".join(ls)


def diff_texts(a, b, name_a="a", name_b="b"):
    """Unified diff between two texts (deterministic, difflib), bounded to MAX_DIFF_LINES."""
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                      fromfile=name_a, tofile=name_b, lineterm=""))
    if not lines:
        return "(no differences)"
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... [diff truncated to {MAX_DIFF_LINES} lines]"]
    return "\n".join(lines)
