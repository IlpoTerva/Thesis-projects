"""FI -> EN: Finnish operator queries translated to English for semantic search."""
from prompts.common import MQM_TAXONOMY, OUTPUT_RULES, PROTECTED_RULE, render_glossary


PROMPT_VERSION = "fi_en-v2"

SEVERITIES = dict(
    critical="a key technical noun, component, alarm code or the problem itself is lost or wrong, "
             "so a search with this query would retrieve the wrong manual sections",
    major="the meaning is distorted or a domain term is replaced with a non-standard one, "
          "retrieval would probably still work but less reliably",
    minor="wording, grammar or style problems that do not affect meaning or retrieval",
)

SYSTEM = """You are an expert evaluator of Finnish-to-English translations in the domain of CNC machine
operation and maintenance (Siemens SINUMERIK 808D). You annotate translation errors using the MQM framework.

The texts are short questions typed by Finnish shopfloor operators. The English translation is used as a
query for semantic search over English service and programming manuals. Accuracy of technical terms and
of the operator's intent matters most. Fluency and style matter little.

{taxonomy}

{protected}
{glossary}
{rules}"""

REFERENCE_BASED = """Evaluate the candidate translation of the Finnish source.
The reference is a manually made golden translation that shows the correct terminology and intent.

For every error, fill `reference_span` with the exact words copied from the reference that correspond
to the erroneous part, i.e. what the candidate should have said. Copy them verbatim from <reference_en>.
If the reference has no corresponding words (e.g. for an addition error), use an empty string.

<source_fi>
{source}
</source_fi>

<reference_en>
{ground}
</reference_en>

<candidate_en>
{output}
</candidate_en>"""

REFERENCE_FREE = """Evaluate the candidate translation of the Finnish source.
There is no reference translation, so leave `reference_span` as an empty string for every error.

<source_fi>
{source}
</source_fi>

<candidate_en>
{output}
</candidate_en>"""


def system_prompt(glossary: bool) -> str:
    return SYSTEM.format(
        taxonomy=MQM_TAXONOMY.format(**SEVERITIES),
        protected=PROTECTED_RULE,
        glossary=render_glossary(glossary),
        rules=OUTPUT_RULES,
    )


def user_prompt(source: str, output: str, ground: str = "") -> str:
    if ground.strip():
        return REFERENCE_BASED.format(source=source, ground=ground, output=output)
    return REFERENCE_FREE.format(source=source, output=output)