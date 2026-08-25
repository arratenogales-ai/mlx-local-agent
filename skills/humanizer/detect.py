#!/usr/bin/env python3
"""detect.py: DETERMINISTIC detector of mechanical AI tells in text (Spanish).

Core piece of the "humanizer" skill. It flags the tells you should not leave to the 14B model
(em dashes, smart quotes, emoji, AI vocabulary, "rule of three", negative parallelism
"no solo... sino...", filler words, title case, lists with bold lead-ins).
It returns EXACT findings `[line, pattern, suggestion]`; the rewrite is done by the model, guided
by those findings. Detection is deterministic, so the model never decides what counts as a tell.

Usage:
    python detect.py text.txt          # readable report
    python detect.py < text.txt        # from stdin
    python detect.py text.txt --json   # JSON [{line,pattern,suggestion,snippet}, ...]

Safety: LINEAR regexes (no catastrophic backtracking) + size caps + timeout (SIGALRM, anti-ReDoS).
It runs as an isolated script under the agent's guard/tooling; aimed at SPANISH, fully local, no
external dependencies.

Note on the data below: the vocabulary lists, the regexes and the labels quoting matched text are
Spanish on purpose. They are the linguistic data the tool analyses, not prose, so they stay Spanish.
"""
import json
import re
import sys

MAX_CHARS = 200_000        # text cap (a huge input must not blow up or hang detection)
MAX_LINE = 4_000           # per-line cap (bounds the regex cost per line)
TIMEOUT_SECONDS = 8        # anti-ReDoS backstop (SIGALRM); the regexes are already linear
MAX_SNIPPET = 80           # how much of the matching line is quoted in each finding

# AI vocabulary (Spanish): stock openers and grandiose words typical of generated text.
# High-signal phrases only, so we do not over-flag ordinary prose.
AI_VOCAB = [
    "en el vasto mundo de", "en la era de", "en el panorama actual", "en el mundo actual",
    "un mundo cada vez más", "hoy en día", "a lo largo de la historia",
    "es importante destacar", "es importante mencionar", "es importante señalar",
    "cabe destacar", "cabe mencionar", "cabe señalar", "cabe resaltar",
    "en resumen", "en conclusión", "en última instancia",
    "profundizar en", "adentrarse en", "sumergirse en", "navegar por",
    "desbloquear", "desatar el potencial", "aprovechar el poder", "empoderar",
    "en el ámbito de", "en el contexto de", "de la mano de", "ir más allá",
    "un testimonio de", "un reflejo de", "no es más que", "piedra angular",
    "en constante evolución", "el panorama", "revolucionar", "fomentar",
    "de vital importancia", "juega un papel crucial", "juega un papel fundamental",
    "sin más preámbulos", "dicho esto",
]

# Filler words (Spanish): padding adverbs and connectives.
FILLERS = [
    "básicamente", "obviamente", "simplemente", "de hecho", "en realidad",
    "sin duda", "sin lugar a dudas", "por así decirlo", "en cierto modo",
    "de alguna manera", "a decir verdad", "en definitiva", "por otro lado",
    "por su parte", "cabe la posibilidad",
]


def _alternation(phrases):
    """LINEAR regex: alternation of escaped literals, with boundaries that avoid nested
    quantifiers (anti-ReDoS). `(?<!\\w)...(?!\\w)` prevents splitting words without backtracking."""
    ordered = sorted(phrases, key=len, reverse=True)   # longest first, for the most specific match
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(r"(?<!\w)(" + body + r")(?!\w)", re.IGNORECASE)


RX_VOCAB = _alternation(AI_VOCAB)
RX_FILLER = _alternation(FILLERS)
RX_DASH = re.compile("[\\u2014\\u2013]")  # em dash, en dash (escaped so this file has none)
RX_SMART_QUOTE = re.compile("[“”‘’«»]")
RX_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF\U00002190-\U000021FF\U0000FE00-\U0000FE0F\U00002700-\U000027BF]")
# Spanish "no solo ... sino ..." and variants. BOUNDED, lazy quantifier, so it stays linear.
RX_NEG = re.compile(r"\bno\s+s[oó]lo\b[^.\n]{0,120}?\bsino\b", re.IGNORECASE)
RX_NEG2 = re.compile(r"\bno\s+se\s+trata\s+de\b[^.\n]{0,120}?\bsino\b", re.IGNORECASE)
# Spanish triad of single words, e.g. "rápido, eficiente y potente" (possible "rule of three").
RX_TRIAD = re.compile(r"\b[\wáéíóúñ]+,\s+[\wáéíóúñ]+\s+y\s+[\wáéíóúñ]+\b", re.IGNORECASE)
# List with a bold lead-in: "- **Punto:** ..." or "- **Punto**: ..." (colon inside or outside the **).
RX_BOLD_LIST = re.compile(r"^\s*[-*+]\s+\*\*[^*\n]{1,80}\*\*")
RX_CAPITALIZED_WORD = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+$")


def _snippet(line, start):
    """Quoted fragment around the match (trimmed)."""
    t = line.strip()
    return t[:MAX_SNIPPET] + ("…" if len(t) > MAX_SNIPPET else "")


def _is_title_case(line):
    """Does the line look like a Title With Every Word Capitalized? (unnatural in Spanish)."""
    clean = line.strip().lstrip("#").strip()
    if not clean or clean.endswith((".", ":", ";", ",")):
        return False
    words = clean.split()
    if len(words) < 3 or len(words) > 12:
        return False
    caps = sum(1 for w in words if RX_CAPITALIZED_WORD.match(w))
    return caps >= 3 and caps / len(words) >= 0.6


def detect(text):
    """Scan `text` and return a list of findings: {line, pattern, suggestion, snippet}.
    Deterministic and bounded. Line by line, which caps the cost of each regex."""
    text = (text or "")[:MAX_CHARS]
    findings = []

    def add(line_no, pattern, suggestion, line, pos=0):
        findings.append({"line": line_no, "pattern": pattern,
                         "suggestion": suggestion, "snippet": _snippet(line, pos)})

    for i, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE:
            line = line[:MAX_LINE]
        if RX_DASH.search(line):
            add(i, "em dash", "use a comma, parentheses, or split into two sentences", line)
        if RX_SMART_QUOTE.search(line):
            add(i, "smart quotes", "use straight quotes \" or drop them", line)
        if RX_EMOJI.search(line):
            add(i, "emoji", "remove it: it has no place in formal prose", line)
        for m in RX_VOCAB.finditer(line):
            add(i, f"AI vocabulary: «{m.group(1)}»", "say it directly and concretely, no formula", line)
        for m in RX_FILLER.finditer(line):
            add(i, f"filler: «{m.group(1)}»", "delete it: it adds no information", line)
        if RX_NEG.search(line) or RX_NEG2.search(line):
            add(i, "negative parallelism «no solo... sino...»", "rephrase as positive and direct", line)
        if RX_TRIAD.search(line):
            add(i, "possible «rule of three» (triad)", "vary the rhythm: do not chain triplets", line)
        if RX_BOLD_LIST.search(line):
            add(i, "list with a bold lead-in («**Punto:**»)", "fold it into prose or simplify it", line)
        if _is_title_case(line):
            add(i, "title case", "in Spanish only the first word is capitalized", line)
    return findings


def _format(findings):
    if not findings:
        return "no mechanical AI tells found"
    parts = [f"L{f['line']} · {f['pattern']} · {f['suggestion']}  «{f['snippet']}»"
             for f in findings]
    parts.append(f"\n{len(findings)} mechanical tell(s) found.")
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
    # Anti-ReDoS backstop: if the scan somehow runs long, abort cleanly. The regexes are already
    # linear, so this is just a seatbelt.
    try:
        import signal
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(TIMEOUT_SECONDS)
    except (ImportError, ValueError, AttributeError):
        pass  # no SIGALRM available (e.g. not the main thread): the linear regexes are enough
    text = _read_input(argv)
    try:
        findings = detect(text)
    except TimeoutError:
        print("ERROR: detection aborted by timeout (anomalous text)", file=sys.stderr)
        return 3
    if "--json" in argv:
        print(json.dumps(findings, ensure_ascii=False))
    else:
        print(_format(findings))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
