---
name: translate
when_to_use: translate, translate this, translate it for me, translation, translate to english, translate to spanish, translate this to english, put it in english, put it in spanish, turn it into english, english version, how do you say in english, how do you say in spanish, translate this text
---

# Skill: translate (ES / EN)

Translates the user's text between **Spanish and English**, in whichever direction they ask for (or into
the opposite language of the text if they don't say). This is a MODEL task: there is no mechanical part.

## Rules

1. **Faithful meaning.** Translate the sense, not word by word. Don't add or drop information. Keep
   numbers, proper nouns and units.
2. **Direction.** If the user names a target language ("to English", "to Spanish"), follow it. If not,
   translate into the **other** language relative to the text (ES to EN, or EN to ES).
3. **Register and tone.** Keep the register of the original (formal/informal, technical/colloquial).
   Don't "improve" it or dress it up.
4. **Terminology.** Use the correct technical term in the target language; when in doubt, the most
   standard one. Don't invent translations for proper nouns or brand names.
5. **Formatting.** Preserve the original line breaks, lists and punctuation.
6. **No meta-commentary.** Return **only the translation**, with no translator's notes unless the user
   asks for them. No "Here is the translation:".

## Example

**Input (ES):** «El sistema procesa 10.000 peticiones por segundo sin caídas.»
**Output (EN):** «The system processes 10,000 requests per second without downtime.»
