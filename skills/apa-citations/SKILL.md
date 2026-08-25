---
name: apa-citations
when_to_use: APA, check my citations, review my citations, APA format, APA style, APA rules, check the citation format, review the references, check the bibliography, are these citations correct, is this reference correct, cite in APA, bibliography format, check APA references, review APA bibliography
script: review_apa.py
---

# Skill: citation review / APA

Reviews the format of **citations and references in APA style (7th edition)** in Spanish-language
texts. The MECHANICAL part is detected by `review_apa.py` (exact findings are injected into your
context); you explain and correct them, **without inventing** data that is not there (if the year is
missing, say so; do not make one up).

## Rules

1. **Fix the format only**, not the content. If a piece of data is missing (year, author),
   **point it out**; never invent a year, an author or a DOI.
2. **Apply the script findings** (each one comes with its line number): citation without a comma
   `(Author, year)`, `et al.` with a period, author initial with a period `Pérez, J.`, reference
   with no year, and so on.
3. **Useful APA 7 reminders:** authors as surname + initial (`Pérez, J.`); year in parentheses after
   the author; article title in lowercase (except the first letter and proper nouns); journal name in
   italics; use `&` inside parentheses and `y` in narrative text (the texts are in Spanish);
   `et al.` from 3 authors onwards.
4. **Output format:** first a **short list** of the problems found (with the line number and the
   fix), and then, if the user asks for it, the **corrected version** of the references.
5. **Honesty:** this is a **format** review, not a check that the source exists or is real. Do not
   verify whether the citation is truthful; that cannot be done without access to the sources.

## Example

**Input:** `Perez, J (2020). Un estudio. Revista de Cosas.`
**Problems:** author initial without a period (`J.`); missing comma before the year, so `(2020)` is
fine here but the in-text citation would be `(Pérez, 2020)`.
**Fixed:** `Pérez, J. (2020). Un estudio. Revista de Cosas.`
