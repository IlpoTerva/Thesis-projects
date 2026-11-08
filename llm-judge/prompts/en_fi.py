"""EN -> FI: English assistant answers translated to Finnish for the operator."""
from prompts.common import MQM_TAXONOMY, OUTPUT_RULES, PROTECTED_RULE, render_glossary

PROMPT_VERSION = "en_fi-v1"

SEVERITIES = dict(
    critical="the operator would be misled: a wrong instruction, step, value, code or component, "
             "a missing step or warning, or text that is unintelligible",
    major="the meaning is distorted, a technical term is wrong or non-standard Finnish, "
          "or grammar errors make a sentence hard to understand",
    minor="small grammar, inflection, spelling or style problems; the meaning is still clear",
)

SYSTEM = """You are an expert evaluator of English-to-Finnish translations in the domain of CNC machine
operation and maintenance (Siemens SINUMERIK 808D). You annotate translation errors using the MQM framework.

The texts are answers from a maintenance assistant, translated into Finnish for shopfloor operators.
Check especially:
- that every instruction and step is translated correctly and nothing is left out or added
- correct Finnish technical terminology (not word-by-word literal coinages)
- Finnish grammar: case endings, compounding, agreement and natural word order
- that numbering, bullet points, markdown and code blocks are kept intact

{taxonomy}

{protected}
{glossary}
{rules}"""

USER = """Evaluate the Finnish translation of the English source.

<source_en>
{source}
</source_en>

<candidate_fi>
{output}
</candidate_fi>"""


def system_prompt(glossary: bool) -> str:
    return SYSTEM.format(
        taxonomy=MQM_TAXONOMY.format(**SEVERITIES),
        protected=PROTECTED_RULE,
        glossary=render_glossary(glossary),
        rules=OUTPUT_RULES,
    )


def user_prompt(source: str, output: str, ground: str = "") -> str:
    return USER.format(source=source, output=output)
