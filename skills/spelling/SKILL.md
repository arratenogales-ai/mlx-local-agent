---
name: spelling
when_to_use: spellcheck, check my spelling, fix typos, proofread, grammar check, fix the spelling, correct spelling mistakes, correct grammar mistakes, spelling errors, grammar errors, fix the accents, check the accents, proofread this text, review spelling and grammar
script: review.py
---

# Skill: spelling and grammar correction

Corrects the **spelling and grammar** of the user's text. This skill targets **Spanish text**: the
rules, word patterns and dictionaries it uses are Spanish. The `review.py` script does the exact
part in two layers:

- The **mechanical, safe** layer (always, no dictionary needed): missing opening `¿`/`¡` marks,
  double spaces, space before punctuation, repeated words, repeated marks.
- **Word-level spelling** with **hunspell** (a real spellchecker with a Spanish dictionary) **if it
  is installed**. It gives the exact list of misspelled words. **It is optional:** with no hunspell
  the script **degrades honestly** to the mechanical layer and says so, and word-level spelling
  (accents, letters) is then handled by the **model**, best-effort (it is a 14B model, so something
  may slip through; **do not promise a perfect correction**). To install it:
  `brew install hunspell` (macOS) or `apt install hunspell hunspell-es` (Linux), with an `es_ES`
  dictionary.

## Rules

1. **Apply the script's findings** (each one comes with its line number): they are safe mechanical
   errors.
2. **Correct the spelling and grammar** of the rest as well as you can: accents, agreement
   (gender/number), verb tenses, punctuation. **Do not change the meaning or the style**; correct,
   do not rewrite.
3. **Preserve** formatting, proper nouns, technical terms and literal quotes.
4. **Be honest:** if you are unsure about a word, do not "invent" a correction; better to leave it
   and, if useful, flag it with a short note.
5. **Output format:** the **corrected text**. If the user asks for the list of changes, add it
   afterwards. No filler meta-commentary.

## Example

**Input:** «Como estas ? Espero que  todo vaya vaya bien y que ayas descansado.»
**Corrected:** «¿Cómo estás? Espero que todo vaya bien y que hayas descansado.»
(mechanical: missing `¿`, space before `?`, double space, repeated "vaya vaya"; model-level
spelling: "ayas" -> "hayas".)
