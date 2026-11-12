from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "accuracy/mistranslation",
    "accuracy/omission",
    "accuracy/addition",
    "accuracy/untranslated",
    "terminology",
    "fluency/grammar",
    "fluency/spelling",
    "fluency/punctuation",
    "style",
    "format/protected_token",
    "format/structure",
    "non_translation",
]
Severity = Literal["minor", "major", "critical"]
Direction = Literal["fi-en", "en-fi"]
Mode = Literal["reference_based", "reference_free"]
Status = Literal["ok", "judge_error", "skipped_wrong_source_lang", "skipped_empty_source"]


class ErrorSpan(BaseModel):
    span: str = Field(description="Exact substring of the translation that contains the error, or empty string for omissions")
    source_span: str = Field(description="The corresponding part of the source text")
    reference_span: str = Field(
        "",
        description="Only when a reference translation is given: the exact words of the reference that "
                    "correspond to this error, i.e. the correct translation. Empty string otherwise.",
    )
    category: Category
    severity: Severity
    explanation: str = Field(description="One short sentence why this is an error")
    suggested_fix: str = Field(description="Corrected text for the span")


class MQMAnnotation(BaseModel):
    """What the judge LLM returns. Scores are computed in code, never by the LLM."""
    errors: list[ErrorSpan] = Field(description="All errors found; empty list if the translation is perfect")
    overall_comment: str = Field(description="One or two sentences summarising translation quality")


class JudgedError(ErrorSpan):
    origin: Literal["llm", "rule"] = "llm"


class ItemResult(BaseModel):
    id: str
    index: int
    direction: Direction
    eval_model: str = "" 
    mode: Mode
    status: Status
    source: str
    output: str
    ground: str = ""

    errors: list[JudgedError] = []
    penalty: float | None = None
    score: float | None = None
    passed: bool | None = None
    has_critical: bool | None = None
    sample_penalties: list[float] = []
    overall_comment: str = ""

    chrf: float | None = None
    bleu: float | None = None

    model: str  # judge model
    base_url: str
    prompt_version: str
    prompt_hash: str
    glossary: bool
    timestamp: str
    latency_s: float = 0.0
    raw: list[str] = []