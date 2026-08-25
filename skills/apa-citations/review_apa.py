#!/usr/bin/env python3
"""review_apa.py: DETERMINISTIC detector of APA (7th ed.) formatting errors in Spanish texts.

Critical part of the "apa-citations" skill: it flags MECHANICAL formatting errors (the ones best not
left to the 14B model) and returns `[line, pattern, suggestion]`. It does NOT check whether the
source is real (that cannot be done without access to the sources) and it does not fix content: only
FORMAT. The model does the correction/explanation, guided by these findings.

Usage:
    python review_apa.py refs.txt          # readable report
    python review_apa.py < refs.txt        # via stdin
    python review_apa.py refs.txt --json   # JSON

Security: LINEAR regexes (no catastrophic backtracking) + size cap + timeout (SIGALRM, anti-ReDoS).
Targets SPANISH/APA, 100% local, no dependencies.
"""
import json
import re
import sys

MAX_CHARS = 200_000
MAX_LINE = 4_000
TIMEOUT_SEC = 8
MAX_FRAG = 90

# The patterns below are LINGUISTIC DATA for SPANISH text (accented letters, "s.f." for "no date",
# Spanish author conventions). Keep them in Spanish: they must match the language being reviewed.

# A plausible year (1500-2099) or "s.f." (no date). Linear.
RX_ANYO = re.compile(r"\((?:19\d\d|20\d\d|1[5-8]\d\d|s\.?\s?f\.?)[a-z]?\)")
RX_ANYO_SUELTO = re.compile(r"\b(?:19\d\d|20\d\d)\b")
# A line that "looks like" a reference/author: starts with "Surname, X" (uppercase initial).
RX_AUTOR_INI = re.compile(r"^[A-ZÁÉÍÓÚÑ][\wáéíóúüñ'’-]+,\s+[A-ZÁÉÍÓÚÑ]")
# Author initial WITHOUT a period: ", J" not followed by a period or another letter (not a full name).
RX_INICIAL_SIN_PUNTO = re.compile(r",\s+[A-ZÁÉÍÓÚÑ](?![.\wáéíóúüñ])")
# In-text citation WITHOUT a comma: "(Author 2020)" -> APA wants "(Author, 2020)". Linear.
RX_CITA_SIN_COMA = re.compile(r"\([A-ZÁÉÍÓÚÑ][\wáéíóúüñ.&\s]{0,40}?\s\d{4}[a-z]?\)")
# "et al" without a period.
RX_ETAL = re.compile(r"\bet\s+al\b(?!\.)", re.IGNORECASE)
# "and" instead of "&"/"y" between authors (APA in Spanish).
RX_AND = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\wáéíóúüñ'’-]+\s+and\s+[A-ZÁÉÍÓÚÑ]")


def _frag(line):
    t = line.strip()
    return t[:MAX_FRAG] + ("…" if len(t) > MAX_FRAG else "")


def detect(text):
    """Scan `text` and return APA formatting findings: {line, pattern, suggestion, fragment}."""
    text = (text or "")[:MAX_CHARS]
    findings = []

    def add(n, pattern, suggestion, line):
        findings.append({"line": n, "pattern": pattern, "suggestion": suggestion,
                         "fragment": _frag(line)})

    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE:
            line = line[:MAX_LINE]
        if not line.strip():
            continue
        # Reference (starts with "Surname, Initial") but with no year in any form: date is missing.
        if RX_AUTOR_INI.search(line) and not RX_ANYO.search(line) and not RX_ANYO_SUELTO.search(line):
            add(i, "reference with no year", 'add the date: Apellido, N. (año). …', line)
        if RX_INICIAL_SIN_PUNTO.search(line):
            add(i, "author initial without a period", 'the initial takes a period: "Pérez, J."', line)
        if RX_CITA_SIN_COMA.search(line):
            add(i, "citation without a comma", 'in APA a citation is "(Autor, año)", with a comma', line)
        if RX_ETAL.search(line):
            add(i, '"et al" without a period', 'it is written "et al." (with a period)', line)
        if RX_AND.search(line):
            add(i, '"and" between authors', 'in Spanish use "y" (narrative) or "&" (in parentheses), not "and"', line)
    return findings


def _format(findings):
    if not findings:
        return "no APA formatting errors detected"
    parts = [f"L{f['line']} · {f['pattern']} · {f['suggestion']}  \"{f['fragment']}\"" for f in findings]
    parts.append(f"\n{len(findings)} possible APA formatting error(s).")
    return "\n".join(parts)


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
        signal.alarm(TIMEOUT_SEC)
    except (ImportError, ValueError, AttributeError):
        pass
    text = _read_input(argv)
    try:
        findings = detect(text)
    except TimeoutError:
        print("ERROR: detection aborted by timeout (anomalous text)", file=sys.stderr)
        return 3
    print(json.dumps(findings, ensure_ascii=False) if "--json" in argv else _format(findings))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
