"""Deterministic checks that run before the LLM judge."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from lingua import Language, LanguageDetectorBuilder

from schemas import JudgedError


LANG_DETECTOR = LanguageDetectorBuilder.from_languages(Language.FINNISH, Language.ENGLISH).build()
_LANGS = {"fi": Language.FINNISH, "en": Language.ENGLISH}
# Short queries are hard to detect, so only act on confident detections.
_LANG_CONFIDENCE = 0.9

# Copied from multilingual-assistant/app/chat_lang.py (_PROTECT_RE) so the judge checks the same tokens
# the translator is told to keep. Importing chat_lang would build the LLMs and the graph.
PROTECT_RE = re.compile(
    r"\b(?:MD\d+(?:\[\d+\])?|X\d{1,3}|DB\d+\.DB[XB]\d+(?:\.\d+)?|[AF]\d{2,5}|\d{4,6})\b|\[\d+\]"
)
_CITATION_RE = re.compile(r"\[\d+\]")


@dataclass
class CheckResult:
    skip_status: str | None = None   # set -> item is not judged at all
    skip_llm: bool = False           # rule errors are conclusive, no need to call the judge
    errors: list[JudgedError] = field(default_factory=list)


def is_language(text: str, lang: str) -> bool:
    """True when the detector is confident `text` is in `lang`."""
    return LANG_DETECTOR.compute_language_confidence(text, _LANGS[lang]) >= _LANG_CONFIDENCE


def _rule_error(category: str, severity: str, span: str, source_span: str, explanation: str) -> JudgedError:
    return JudgedError(span=span, source_span=source_span, category=category, severity=severity,
                       explanation=explanation, suggested_fix=source_span, origin="rule")


def missing_protected_tokens(source: str, output: str) -> list[str]:
    out_tokens = set(PROTECT_RE.findall(output))
    return sorted(t for t in set(PROTECT_RE.findall(source)) if t not in out_tokens)


def run_checks(source: str, output: str, src_lang: str, tgt_lang: str) -> CheckResult:
    res = CheckResult()
    if not source.strip():
        res.skip_status = "skipped_empty_source"
        return res
    if is_language(source, tgt_lang):
        # e.g. the answer was already in Finnish before the EN->FI step: nothing was really translated
        res.skip_status = "skipped_wrong_source_lang"
        return res

    if not output.strip():
        res.errors.append(_rule_error("accuracy/omission", "critical", "", source, "Translation is empty."))
        res.skip_llm = True
        return res
    if output.strip() == source.strip():
        res.errors.append(_rule_error("accuracy/untranslated", "critical", output, source,
                                      "Output is identical to the source."))
        res.skip_llm = True
        return res
    if is_language(output, src_lang):
        res.errors.append(_rule_error("accuracy/untranslated", "critical", output, source,
                                      f"Output is in the source language ({src_lang}), not {tgt_lang}."))

    for tok in missing_protected_tokens(source, output):
        severity = "minor" if _CITATION_RE.fullmatch(tok) else "critical"
        res.errors.append(_rule_error("format/protected_token", severity, "", tok,
                                      f"Protected token '{tok}' is missing or changed in the translation."))
    return res


def merge_errors(rule_errors: list[JudgedError], llm_errors: list[JudgedError]) -> list[JudgedError]:
    """Drop LLM protected-token errors that a rule already reported for the same token."""
    flagged = {e.source_span for e in rule_errors if e.category == "format/protected_token"}
    kept = [
        e for e in llm_errors
        if not (e.category == "format/protected_token"
                and any(tok in e.source_span or tok in e.span for tok in flagged))
    ]
    return rule_errors + kept
