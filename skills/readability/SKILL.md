---
name: readability
when_to_use: readability, how readable is this, reading level, readability score, is this text hard to read, measure readability, readability metrics, Fernandez-Huerta, INFLESZ, Szigriszt, Flesch
script: measure.py
---

# Skill: measure the readability of a text (deterministic metrics)

**Measures** how readable a Spanish text is and **suggests concrete improvements**. The NUMBERS come
from the `measure.py` script (exact and reproducible); you **interpret** them and propose changes.
This is the counterpart to the *humanizer* skill: one **measures**, the other **rewrites**.

The script returns: words, sentences, syllables (estimated), **average sentence length** (words),
**average word length** (syllables/characters), **percentage of long sentences** (>30 words),
**estimated periphrastic passive voice** (%), and two indices: **Fernandez-Huerta** and **INFLESZ
(Szigriszt-Pazos)** with its band (very difficult to very easy). Both indices are Spanish adaptations
of Flesch and only score SPANISH text; running them on English input produces meaningless numbers.

## Rules

1. **Start from the script's numbers** (they are the measurable truth). Do not recompute them by eye
   and do not invent them.
2. **Interpret with judgment and honesty:**
   - High INFLESZ/Fernandez-Huerta (>65) means an easy text; low (<55) means dense or technical.
   - Long sentences and heavy passive voice usually **lower** readability and clarity.
   - **The syllable count is an estimate** (vowel groups) and so is the passive voice detection: say
     so if the user wants precision. Do not sell it as 100% exact.
3. **Suggest concrete, actionable improvements** anchored to the numbers, for example: "40% of your
   sentences run past 30 words, split the longest ones"; "there is a lot of passive voice, switch it
   to active where you can".
4. **Do not rewrite the whole text** unless asked: the focus here is **measuring and advising**. If
   the user wants a full "more human" rewrite, that is the *humanizer* skill.
5. **Be brief and useful:** numbers first, then 2 to 4 prioritized suggestions. No filler.

## Example

**Script numbers (example):** average sentence length 34 words, 45% long sentences, estimated passive
voice 30%, INFLESZ 48 (somewhat difficult).

**Interpretation:** the text is **dense**. Very long sentences and a fair amount of passive voice make
it costly to read (INFLESZ 48 = "somewhat difficult"). **Suggestions:** (1) split sentences over 30
words in two; (2) switch to active voice ("se analizaron los datos" becomes "analizamos los datos");
(3) aim for an INFLESZ of 55 to 65 for a general audience.
