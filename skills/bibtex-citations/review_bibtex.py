#!/usr/bin/env python3
"""review_bibtex.py: validates and normalises BibTeX entries DETERMINISTICALLY.

Critical part of the "bibtex-citations" skill: it parses `@type{key, field = {value}, ...}` entries
and detects the **key**, the **fields present** and the **REQUIRED fields that are missing** for that
type (standard BibTeX). It also flags malformed entries (unclosed, no key) and non-standard types.
The script gives the exact diagnosis; the model builds or fixes the entries guided by it.

Counterpart to the "apa-citations" skill: APA covers the TEXT, BibTeX covers the reference MANAGER.

Usage:
    python review_bibtex.py refs.bib          # readable report
    python review_bibtex.py < refs.bib        # from stdin
    python review_bibtex.py refs.bib --json    # JSON

Safety: parsing by brace counting (NO regex over the whole text) + size cap + timeout
(SIGALRM, anti-ReDoS). 100% local, no dependencies.
"""
import json
import sys

MAX_CHARS = 500_000
TIMEOUT_SECONDS = 8

# REQUIRED fields per entry type (standard BibTeX). Each requirement is a list of
# ALTERNATIVES (one is enough): for example, a @book needs author OR editor.
REQUIRED = {
    "article":       [["author"], ["title"], ["journal"], ["year"]],
    "book":          [["author", "editor"], ["title"], ["publisher"], ["year"]],
    "booklet":       [["title"]],
    "inbook":        [["author", "editor"], ["title"], ["chapter", "pages"], ["publisher"], ["year"]],
    "incollection":  [["author"], ["title"], ["booktitle"], ["publisher"], ["year"]],
    "inproceedings": [["author"], ["title"], ["booktitle"], ["year"]],
    "conference":    [["author"], ["title"], ["booktitle"], ["year"]],
    "manual":        [["title"]],
    "mastersthesis": [["author"], ["title"], ["school"], ["year"]],
    "phdthesis":     [["author"], ["title"], ["school"], ["year"]],
    "proceedings":   [["title"], ["year"]],
    "techreport":    [["author"], ["title"], ["institution"], ["year"]],
    "unpublished":   [["author"], ["title"], ["note"]],
    "misc":          [],
}
# @types that are NOT references: they are ignored (not validated as bibliographic entries).
_NOT_AN_ENTRY = {"string", "comment", "preamble", "set"}


def _blocks(text):
    """Split the text into (type, body, closed) for each `@type{ ... }`, with robust brace counting
    (tolerates nested braces inside values). `closed` = the closing brace was found."""
    blocks = []
    i, n = 0, len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        j = at + 1
        while j < n and text[j].isalpha():
            j += 1
        entry_type = text[at + 1:j].lower()
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j >= n or text[j] != "{":
            i = at + 1
            continue
        depth, k, closed, in_quotes = 0, j, False, False
        while k < n:
            c = text[k]
            if in_quotes:                             # inside a "..." value, braces and commas are text
                if c == '"' and text[k - 1] != "\\":
                    in_quotes = False
            elif c == '"' and depth == 1:             # start of a quoted value (field level)
                in_quotes = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            k += 1
        blocks.append((entry_type, text[j + 1:k], closed))
        i = (k + 1) if closed else (j + 1)
    return blocks


def _key_and_fields(body):
    """From `key, field = value, ...` extract (key, {field: value}) splitting on LEVEL 0 commas."""
    parts, buf, depth, in_quotes, prev = [], [], 0, False, ""
    for c in body:
        if in_quotes:                                 # a comma inside "..." does NOT separate fields
            if c == '"' and prev != "\\":
                in_quotes = False
        elif c == '"' and depth == 0:
            in_quotes = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "," and depth == 0 and not in_quotes:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        prev = c
    parts.append("".join(buf))
    key = parts[0].strip()
    fields = {}
    for p in parts[1:]:
        if "=" in p:
            k, _, _v = p.partition("=")
            name = k.strip().lower()
            if name:
                fields[name] = True
    return key, fields


def validate(text):
    """Validate every BibTeX entry in `text`. Returns {entries:[...], n_entries, n_with_issues}."""
    text = (text or "")[:MAX_CHARS]
    entries = []
    for entry_type, body, closed in _blocks(text):
        if entry_type in _NOT_AN_ENTRY:
            continue
        key, fields = _key_and_fields(body)
        warnings, missing = [], []
        if not closed:
            warnings.append('unclosed entry (missing "}")')
        if not key or "=" in key or " " in key.strip():
            warnings.append("empty or invalid key (citekey)")
        if entry_type not in REQUIRED:
            warnings.append(f"non-standard type: @{entry_type}")
        for req in REQUIRED.get(entry_type, []):
            if not any(alt in fields for alt in req):
                missing.append("/".join(req))
        entries.append({
            "type": entry_type, "key": key, "fields": sorted(fields.keys()),
            "missing": missing, "warnings": warnings,
        })
    n_issues = sum(1 for e in entries if e["missing"] or e["warnings"])
    return {"entries": entries, "n_entries": len(entries), "n_with_issues": n_issues}


def _format_report(r):
    if not r["entries"]:
        return "no BibTeX entries (@article, @book, ...) found in the text."
    ls = [f"{r['n_entries']} BibTeX entry(ies), {r['n_with_issues']} with issues:"]
    for e in r["entries"]:
        head = f"  @{e['type']}{{{e['key'] or '?'}}}"
        if not e["missing"] and not e["warnings"]:
            ls.append(head + "  OK: complete")
            continue
        ls.append(head + ":")
        if e["missing"]:
            ls.append("      missing required fields: " + ", ".join(e["missing"]))
        for w in e["warnings"]:
            ls.append(f"      warning: {w}")
    return "\n".join(ls)


def _read_input(argv):
    paths = [a for a in argv if not a.startswith("-")]
    if paths:
        try:
            with open(paths[0], encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError as e:
            print(f"ERROR: could not read {paths[0]}: {e}", file=sys.stderr)
            sys.exit(2)
    return sys.stdin.read()


def main(argv):
    try:
        import signal
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(TIMEOUT_SECONDS)
    except (ImportError, ValueError, AttributeError):
        pass
    text = _read_input(argv)
    try:
        r = validate(text)
    except TimeoutError:
        print("ERROR: validation aborted by timeout (anomalous text)", file=sys.stderr)
        return 3
    print(json.dumps(r, ensure_ascii=False) if "--json" in argv else _format_report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
