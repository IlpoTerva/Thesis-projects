"""Shared parts of the judge prompts: MQM taxonomy, severities, glossary, output rules."""

# Expected terminology for the SINUMERIK 808D domain (Finnish -> English).
# The judge treats these as the correct terms. Edit/extend when making golden translations.
GLOSSARY: dict[str, str] = {
    "kara, työkara": "spindle",
    "osaohjelma, osa-ohjelma": "part program",  # NOTE verify: CNC "part program" vs. general "subprogram"
    "aliohjelma": "subprogram / subroutine",
    "herja, herjata (colloquial)": "alarm, to raise an alarm",
    "hälytys": "alarm",
    "vika": "fault",
    "virhe": "error",
    "työkalu": "tool",
    "syöttö": "feed",
    "akseli": "axis",
    "hätäseis, hätä-seis": "emergency stop",
    "ohjelmanhallinta": "program manager",
    "käyttöpaneeli": "operator panel",
}

MQM_TAXONOMY = """Error categories (use exactly these values):
- accuracy/mistranslation: meaning of the source is rendered incorrectly
- accuracy/omission: content of the source is missing from the translation
- accuracy/addition: the translation adds content not in the source
- accuracy/untranslated: source-language text left untranslated where it should be translated
- terminology: a domain term is translated with a wrong, inconsistent or non-standard term
- fluency/grammar: grammar, inflection, word order or agreement errors
- fluency/spelling: spelling or compounding errors
- fluency/punctuation: punctuation errors
- style: awkward, unnatural or overly literal wording that does not change meaning
- format/protected_token: an alarm code, machine data, interface, PLC address, softkey name or citation marker was changed, translated or dropped
- format/structure: numbering, bullet points, line breaks, markdown or code blocks were broken
- non_translation: the output is not a translation of the source at all (e.g. it answers the question instead)

Severity levels:
- critical: {critical}
- major: {major}
- minor: {minor}"""

PROTECTED_RULE = """Protected tokens must appear unchanged in the translation: alarm codes (025000, A08, F30001),
machine data (MD30200, MD14510[20]), interfaces (X54, X102), PLC addresses (DB1000.DBX3.5),
key and softkey names in capitals (RESET, CYCLE START) and citation markers like [1]."""

OUTPUT_RULES = """Annotation rules:
- Annotate every error you find. Each error gets one category and one severity.
- Quote `span` exactly from the translation. For omissions, leave `span` empty and quote the missing part in `source_span`.
- Do NOT penalise typos, spacing or informal language that is already in the source text.
- Different wording than the reference is fine as long as meaning and terminology are correct.
- If the translation has no errors, return an empty `errors` list.
- Do not invent errors to have something to report.
- Respond with JSON only, in this format:
{"errors": [{"span": "...", "source_span": "...", "category": "terminology", "severity": "major",
  "explanation": "...", "suggested_fix": "..."}],
 "overall_comment": "..."}"""


def render_glossary(enabled: bool) -> str:
    if not enabled:
        return ""
    lines = "\n".join(f"- {fi} -> {en}" for fi, en in GLOSSARY.items())
    return f"\nExpected domain terminology (Finnish -> English):\n{lines}\n"
