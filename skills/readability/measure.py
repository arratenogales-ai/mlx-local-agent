#!/usr/bin/env python3
"""measure.py: DETERMINISTIC readability metrics for a Spanish text.

Critical part of the "readability" skill: it gives the exact, reproducible NUMBERS (average sentence
and word length, percentage of long sentences, ESTIMATED periphrastic passive voice, and the
Fernandez-Huerta and INFLESZ/Szigriszt-Pazos indices). The model INTERPRETS them and suggests
concrete improvements; the script does not give opinions.

Counterpart to the "humanizer" skill: this one MEASURES, the other one REWRITES.

Indices (Spanish adaptations of Flesch; both use syllables/word S and words/sentence W). They score
SPANISH text only:
  - Fernandez-Huerta:  L = 206.84 - 60*S - 1.02*W
  - INFLESZ (Szigriszt-Pazos):  P = 206.835 - 62.3*S - W   (scale: <40 very difficult to >80 very easy)
The syllable count is an ESTIMATE (vowel groups; hiatuses are undercounted). This is stated in output.

Usage:
    python measure.py text.txt          # readable report
    python measure.py < text.txt        # via stdin
    python measure.py text.txt --json   # JSON

Safety: LINEAR regexes + size cap + timeout (SIGALRM, anti-ReDoS). 100% local, no dependencies.
"""
import json
import re
import sys

MAX_CHARS = 200_000
TIMEOUT_SECONDS = 8
LONG_SENTENCE_THRESHOLD = 30       # words: above this, a sentence counts as "long" (plain-style rule)

# The data below is Spanish linguistic data on purpose: this skill scores SPANISH text.
# The vowel set, the word character class and the passive-voice markers must stay in Spanish.
_VOWELS = set("aeiouáéíóúýü")
RX_WORD = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)             # word = run of letters (linear)
RX_SENTENCE_END = re.compile(r"[.!?…]+")                          # sentence boundary (linear)
# Estimated Spanish periphrastic passive: a form of "ser" + participle (-ado/-ido). Linear, bounded.
RX_PASSIVE = re.compile(
    r"\b(es|son|fue|fueron|era|eran|será|serán|sido|siendo|fui|fuiste|somos|sois|"
    r"ha\s+sido|han\s+sido|había\s+sido|habían\s+sido)\s+"
    r"[a-záéíóúüñ]{2,}(?:ado|ada|ados|adas|ido|ida|idos|idas)\b", re.IGNORECASE)


def _syllables_in_word(word):
    """ESTIMATED syllables of a word: number of vowel groups (diphthongs count as 1). Minimum 1."""
    groups, prev = 0, False
    for c in word.lower():
        v = c in _VOWELS
        if v and not prev:
            groups += 1
        prev = v
    return max(1, groups)


def _sentences(text):
    """Split into sentences on [.!?…] and newlines. Returns the list of non-empty sentences."""
    chunks = []
    for line in text.split("\n"):
        chunks.extend(RX_SENTENCE_END.split(line))
    return [c.strip() for c in chunks if c.strip()]


def measure(text):
    """Deterministic metrics for `text`. Returns a dict with the exact numbers (or an almost empty
    dict if there are no words). Never raises on normal input."""
    text = (text or "")[:MAX_CHARS]
    words = RX_WORD.findall(text)
    n_words = len(words)
    if n_words == 0:
        return {"words": 0}
    sentences = _sentences(text)
    n_sentences = max(1, len(sentences))
    n_syllables = sum(_syllables_in_word(w) for w in words)
    n_chars = sum(len(w) for w in words)
    # long sentences (by word count)
    long_sentences = sum(
        1 for s in sentences if len(RX_WORD.findall(s)) > LONG_SENTENCE_THRESHOLD)
    # estimated passive voice: sentences with at least one periphrastic passive construction
    passives = sum(1 for s in sentences if RX_PASSIVE.search(s))

    s = n_syllables / n_words                        # syllables per word
    w = n_words / n_sentences                        # words per sentence
    fh = 206.84 - 60.0 * s - 1.02 * w                # Fernandez-Huerta
    inflesz = 206.835 - 62.3 * s - w                 # Szigriszt-Pazos (INFLESZ)
    return {
        "words": n_words,
        "sentences": n_sentences,
        "syllables": n_syllables,
        "avg_sentence_length_words": round(w, 2),
        "avg_word_length_syllables": round(s, 2),
        "avg_word_length_chars": round(n_chars / n_words, 2),
        "long_sentences": long_sentences,
        "pct_long_sentences": round(100.0 * long_sentences / n_sentences, 1),
        "estimated_passive_sentences": passives,
        "pct_estimated_passive": round(100.0 * passives / n_sentences, 1),
        "fernandez_huerta": round(fh, 1),
        "inflesz": round(inflesz, 1),
        "inflesz_band": _inflesz_band(inflesz),
    }


def _inflesz_band(score):
    """Qualitative band of the INFLESZ scale (factual mapping, not an opinion)."""
    if score < 40:
        return "very difficult"
    if score < 55:
        return "somewhat difficult"
    if score < 65:
        return "normal"
    if score < 80:
        return "fairly easy"
    return "very easy"


_NOTE = ("(note: the syllable count is an ESTIMATE based on vowel groups; the passive voice figure "
         "estimates the Spanish periphrastic passive \"ser + participle\". The indices are indicative.)")


def _format(m):
    if not m.get("words"):
        return "no words to measure"
    rows = [
        f"words: {m['words']}  ·  sentences: {m['sentences']}  ·  syllables (est.): {m['syllables']}",
        f"average sentence length: {m['avg_sentence_length_words']} words",
        f"average word length: {m['avg_word_length_syllables']} syllables "
        f"({m['avg_word_length_chars']} characters)",
        f"long sentences (>{LONG_SENTENCE_THRESHOLD} words): {m['long_sentences']} "
        f"({m['pct_long_sentences']}%)",
        f"estimated passive voice: {m['estimated_passive_sentences']} sentences "
        f"({m['pct_estimated_passive']}%)",
        f"Fernandez-Huerta: {m['fernandez_huerta']}   ·   INFLESZ: {m['inflesz']} "
        f"({m['inflesz_band']})",
    ]
    return "\n".join(rows) + "\n" + _NOTE


def _read_input(argv):
    paths = [a for a in argv if not a.startswith("-")]
    if paths:
        try:
            with open(paths[0], encoding="utf-8", errors="replace") as f:
                return f.read()
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
        m = measure(text)
    except TimeoutError:
        print("ERROR: measurement aborted by timeout (anomalous text)", file=sys.stderr)
        return 3
    print(json.dumps(m, ensure_ascii=False) if "--json" in argv else _format(m))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
