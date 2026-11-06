---
name: humanizer
when_to_use: humanize, humanize this, make it sound human, make it more human, make it sound natural, rewrite so it sounds natural, remove the AI tone, take out the AI tells, make it not sound like AI, so it does not read like it was written by an AI, less robotic, drop the robotic tone, give it a more human tone, rewrite it so it does not sound machine written
script: detect.py
---

# Skill: humanizer

Rewrite a text so it **sounds like a person wrote it**, stripping the mechanical tells of AI writing
**without changing what it says**. The mechanical part is found by `detect.py` (the exact findings are
injected for you); your job is to **rewrite** with judgement, keeping facts, figures and meaning intact.

The detector targets **Spanish** text, so the tells below and the examples are Spanish.

## Rules (apply these when rewriting)

1. **Do not change the meaning.** Do not invent or drop facts, figures, names or nuance. If something
   is unclear, leave it as is rather than dressing it up. Keep the original language (Spanish) and
   register.
2. **Remove the tells the script flags:**
   - **Em dashes and en dashes:** replace with a comma, parentheses, or two sentences.
   - **Smart quotes (“ ” « »):** use straight quotes `"` or drop them.
   - **Emoji:** out, in formal prose.
   - **AI vocabulary** ("en el vasto mundo de", "es importante destacar", "cabe mencionar", "en
     constante evolución", "profundizar en"...): say it directly and concretely, no formula.
   - **Filler words** ("básicamente", "obviamente", "sin duda", "en definitiva"...): delete them.
   - **Negative parallelism** ("no solo... sino..."): rephrase as a positive, direct statement.
   - **Rule of three** (triads like "rápido, eficiente y potente"): break the pattern, do not chain
     triplets.
   - **Title Case:** in Spanish only the first word is capitalized.
   - **Lists with bold lead-ins** ("**Punto:** ..."): fold them into prose or simplify them.
3. **Vary the rhythm.** Mix short and long sentences. Do not start them all the same way. Rhythmic
   monotony is what reads as AI more than anything else.
4. **Plain words.** Prefer the simple word to the grand one ("usar" over "aprovechar el poder de";
   "ayuda" over "juega un papel crucial").
5. **Less is more.** Cut preambles and trailing filler. Get to the point.
6. **No meta-commentary.** Return **only the rewritten text**, with no explanation of what you changed
   and no introduction. Nothing like "Aquí tienes el texto humanizado:".

## Examples (before, after)

**Before:** «En el vasto mundo de la programación, es importante destacar que las pruebas no solo
detectan errores, sino que mejoran el diseño 🚀.»
**After:** «Las pruebas detectan errores y, de paso, mejoran el diseño.»

**Before:** «Esta herramienta es rápida, potente y versátil, y sin duda revolucionará tu flujo de
trabajo.»
**After:** «La herramienta es rápida y versátil; te cambiará el flujo de trabajo.»

**Before:** «Cabe mencionar que el proyecto, tras meses de trabajo, finalmente ha llegado a su fin.»
**After:** «Tras meses de trabajo, el proyecto ha terminado.»

If the script flags nothing, still check the **rhythm** and the **grandiose words**: those are the
nuances the detector cannot catch. Preserving the meaning always comes first.
