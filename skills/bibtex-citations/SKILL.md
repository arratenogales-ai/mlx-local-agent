---
name: bibtex-citations
when_to_use: bibtex, my bibtex, the bibtex, in bibtex, bibtex entries, check my bibtex, validate my bibtex, validate references, fix my .bib file, my .bib, the .bib file, bibliography entries, reference manager entries
script: review_bibtex.py
---

# Skill: validate and build BibTeX entries

Validates, normalises and helps build **BibTeX** entries (the ones a reference manager or a `.bib`
file uses). The `review_bibtex.py` script does the exact part: it parses each entry and detects the
**type**, the **key**, the fields present and the **required fields that are missing** for that type.
You explain the results and **complete or fix** the entries guided by its findings.

This is the counterpart to the **APA** skill: APA covers in-**text** citations, BibTeX covers the
**reference manager**.

## Required fields (standard BibTeX), for reference

- `@article`: author, title, journal, year
- `@book`: author or editor, title, publisher, year
- `@inproceedings` / `@conference`: author, title, booktitle, year
- `@incollection`: author, title, booktitle, publisher, year
- `@phdthesis` / `@mastersthesis`: author, title, school, year
- `@techreport`: author, title, institution, year
- `@misc`: nothing required (useful for websites and software; add `howpublished`, `url` or `note`).

## Rules

1. **Start from the script's findings** (missing fields, unclosed entries, non-standard types).
   They are exact; do not recompute them by eye.
2. **Fill in what is missing** by proposing the fields the entry type requires, in the correct
   format (`field = {value}`), and **flag** anything you cannot know (for example, do not invent the
   year or the author: ask for it or leave it marked as `{MISSING}`).
3. **Normalise:** lowercase field names, a citekey with no spaces, balanced braces.
4. **Never invent bibliographic data** (author, year, publisher) that is not there: honesty first.
5. **Output format:** the corrected entries in a BibTeX block, then a short summary of what you
   changed. No filler.

## Example

**Script:** `@article{missing2021}` -> missing required fields: author, year.

**Answer:** "That `@article` entry is missing `author` and `year`, which are required. It would look
like this:"
```bibtex
@article{missing2021,
  author  = {MISSING: add the author(s)},
  title   = {No author and no year},
  journal = {Some Journal},
  year    = {MISSING: add the year}
}
```
"I have not invented them: fill them in yourself."
