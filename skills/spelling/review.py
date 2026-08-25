#!/usr/bin/env python3
"""review.py: DETERMINISTIC detector of spelling mistakes in Spanish text.

Part of the "spelling" skill. Two layers:
  1) MECHANICAL, no dictionary (always): missing opening ¿/¡ marks, double spaces, space before
     punctuation, repeated words, repeated marks.
  2) WORD-LEVEL SPELLING with **hunspell** (a real spellchecker with a Spanish dictionary) IF it is
     installed: returns the list of misspelled words (deterministic). **It is OPTIONAL:** with no
     hunspell (or no Spanish dictionary) it DEGRADES HONESTLY to the mechanical layer and SAYS SO;
     word-level spelling (accents, letters) is then handled by the model, best-effort.

Installing hunspell (optional, fully local and free):
    macOS:  brew install hunspell   plus an es_ES dictionary (e.g. from LibreOffice) in
            ~/Library/Spelling or /Library/Spelling, or export DICPATH to the folder holding
            es_ES.dic/.aff
    Linux:  apt install hunspell hunspell-es   (or your distro's Spanish dictionary package)

Usage:
    python review.py text.txt          # readable report
    python review.py < text.txt        # via stdin
    python review.py text.txt --json   # JSON

Safety: LINEAR regexes plus a size cap plus a timeout (SIGALRM, anti-ReDoS); hunspell runs through
subprocess with a fixed argument list (no shell) and a timeout. Fully local, no Python dependencies.
"""
import json
import re
import shutil
import subprocess
import sys

MAX_CHARS = 200_000
MAX_LINE = 4_000
TIMEOUT_SECONDS = 8
MAX_FRAGMENT = 90
HUNSPELL_TIMEOUT = 8               # cap for the hunspell subprocess (anti-hang)
MAX_MISSPELLED = 200               # cap on reported misspelled words (keeps context light)

# Linguistic data below targets SPANISH text: dictionary names, character classes and the
# punctuation rules are Spanish on purpose. Do not translate them.
_DICTIONARIES = ("es_ES", "es_MX", "es_AR", "es_CO", "es")   # Spanish dictionary candidates

RX_DOUBLE_SPACE = re.compile(r"\S(  +)\S")                     # two or more spaces between words
RX_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%])")          # space before , . ; : ! ? %
RX_REPEATED_WORD = re.compile(r"\b([a-záéíóúüñ]{2,20})\s+\1\b", re.IGNORECASE)  # "de de", "vaya vaya"
RX_REPEATED_MARKS = re.compile(r"([!?])\1|[!?]{2,}|\.{4,}")    # !! ?? ?! ... (4+ dots)


def _fragment(line):
    t = line.strip()
    return t[:MAX_FRAGMENT] + ("…" if len(t) > MAX_FRAGMENT else "")


def detect(text):
    """Scan `text` and return MECHANICAL findings: {line, pattern, suggestion, fragment}."""
    text = (text or "")[:MAX_CHARS]
    findings = []

    def add(n, pattern, suggestion, line):
        findings.append({"line": n, "pattern": pattern, "suggestion": suggestion,
                         "fragment": _fragment(line)})

    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE:
            line = line[:MAX_LINE]
        if not line.strip():
            continue
        # Missing Spanish opening mark: the line has ? (or !) but no ¿ (or ¡).
        if "?" in line and "¿" not in line:
            add(i, "missing opening '¿'", "in Spanish a question opens with '¿'", line)
        if "!" in line and "¡" not in line:
            add(i, "missing opening '¡'", "in Spanish an exclamation opens with '¡'", line)
        if RX_DOUBLE_SPACE.search(line):
            add(i, "double space", "leave a single space between words", line)
        if RX_SPACE_BEFORE_PUNCT.search(line):
            add(i, "space before punctuation", "do not leave a space before , . ; : ! ?", line)
        m = RX_REPEATED_WORD.search(line)
        if m:
            add(i, f"repeated word '{m.group(1)}'", "remove the repetition", line)
        if RX_REPEATED_MARKS.search(line):
            add(i, "repeated marks", "a single mark is enough (more sober register)", line)
    return findings


# --- Word-level spelling with hunspell (optional, honest degradation) ----------
def _hunspell_dictionary():
    """Name of the Spanish hunspell dictionary available (es_ES/es_MX/...), or None if hunspell or
    every Spanish dictionary is missing. Deterministic and bounded."""
    if not shutil.which("hunspell"):
        return None
    for cand in _DICTIONARIES:
        try:
            r = subprocess.run(["hunspell", "-d", cand, "-l"], input="probando\n",
                               capture_output=True, text=True, timeout=HUNSPELL_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            return None
        err = (r.stderr or "").lower()
        if r.returncode == 0 and "can't open" not in err and "no such" not in err:
            return cand
    return None


def hunspell_check(text, dictionary=None):
    """Misspelled words according to hunspell (deduplicated list), or None if hunspell or the
    dictionary is unavailable. `hunspell -l` reads stdin and prints one word per line (no shell)."""
    dictionary = dictionary or _hunspell_dictionary()
    if not dictionary:
        return None
    try:
        r = subprocess.run(["hunspell", "-d", dictionary, "-l"], input=(text or "")[:MAX_CHARS],
                           capture_output=True, text=True, timeout=HUNSPELL_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    seen, misspelled = set(), []
    for w in (r.stdout or "").split():
        if w not in seen:
            seen.add(w)
            misspelled.append(w)
        if len(misspelled) >= MAX_MISSPELLED:
            break
    return misspelled


_NOTE = ("(note: no local spelling dictionary, hunspell is not installed; word-level spelling "
         "(accents, letters) is reviewed by the model best-effort, so something may slip through. "
         "Install hunspell plus a Spanish dictionary for an exact check.)")


def _format_report(findings, dictionary=None, misspelled=None):
    """Readable report. `dictionary`=None means no hunspell (honest note); a dictionary name adds
    the word-level spelling verified by hunspell (`misspelled` = list of misspelled words)."""
    parts = []
    if findings:
        parts += [f"L{h['line']} · {h['pattern']} · {h['suggestion']}  '{h['fragment']}'"
                  for h in findings]
        parts.append(f"\n{len(findings)} mechanical error(s).")
    else:
        parts.append("no mechanical errors detected")
    if dictionary:
        if misspelled:
            parts.append(f"\npossibly misspelled words (hunspell {dictionary}): "
                         + ", ".join(misspelled))
        else:
            parts.append(f"\nword-level spelling: no errors according to hunspell ({dictionary}).")
    else:
        parts.append(_NOTE)
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
        signal.alarm(TIMEOUT_SECONDS)
        _alarm_set = True
    except (ImportError, ValueError, AttributeError):
        _alarm_set = False
    text = _read_input(argv)
    try:
        findings = detect(text)
    except TimeoutError:
        print("ERROR: detection aborted by timeout (anomalous text)", file=sys.stderr)
        return 3
    finally:
        if _alarm_set:                  # disarm the alarm before the hunspell subprocess
            try:
                import signal
                signal.alarm(0)
            except (ImportError, ValueError, AttributeError):
                pass
    dictionary = _hunspell_dictionary()
    misspelled = hunspell_check(text, dictionary) if dictionary else None
    if "--json" in argv:
        print(json.dumps({"findings": findings, "dictionary": dictionary,
                          "misspelled_words": misspelled,
                          "has_local_dictionary": bool(dictionary)}, ensure_ascii=False))
    else:
        print(_format_report(findings, dictionary=dictionary, misspelled=misspelled))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
