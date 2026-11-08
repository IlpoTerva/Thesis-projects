"""LLM judge: one OpenAI-compatible client, structured MQM annotation, retries and sampling."""
from __future__ import annotations
import hashlib
import statistics
import time
from datetime import datetime, timezone
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from checks import merge_errors, run_checks
from config import settings
from metrics import is_pass, mqm_penalty, mqm_score, sentence_bleu, sentence_chrf
from prompts import en_fi, fi_en
from schemas import ItemResult, JudgedError, MQMAnnotation


DIRECTIONS: dict[str, tuple[ModuleType, str, str]] = {
    # direction: (prompt module, source lang, target lang)
    "fi-en": (fi_en, "fi", "en"),
    "en-fi": (en_fi, "en", "fi"),
}


def item_id(source: str, output: str, ground: str = "") -> str:
    return hashlib.sha1(f"{source}\x1f{output}\x1f{ground}".encode("utf-8")).hexdigest()[:16]


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class Judge:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 samples: int | None = None, glossary: bool | None = None):
        self.model = model or settings.JUDGE_MODEL
        self.base_url = base_url or settings.JUDGE_BASE_URL
        self.samples = samples or settings.JUDGE_SAMPLES
        self.glossary = settings.USE_GLOSSARY if glossary is None else glossary
        llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=api_key or settings.JUDGE_API_KEY,
            model=self.model,
            temperature=settings.JUDGE_TEMPERATURE,
            timeout=settings.JUDGE_TIMEOUT,
            max_retries=2,  # transport-level retries (rate limits, timeouts)
        )
        self.llm = llm.with_structured_output(MQMAnnotation, method=settings.JUDGE_STRUCTURED, include_raw=True)

    def annotate(self, system: str, user: str) -> tuple[MQMAnnotation | None, list[str]]:
        """Call the judge until it returns a valid MQMAnnotation. Returns (annotation, raw responses)."""
        msgs: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
        raws: list[str] = []
        for _ in range(settings.JUDGE_MAX_RETRIES):
            try:
                out = self.llm.invoke(msgs)
            except Exception as e:  # provider rejected the request or the output could not be parsed
                raws.append(f"<exception> {type(e).__name__}: {e}")
                continue
            raw = out["raw"].content if out.get("raw") is not None else ""
            raws.append(raw)
            if out.get("parsed") is not None:
                return out["parsed"], raws
            msgs += [AIMessage(content=raw),
                     HumanMessage(content=f"Your response was not valid: {out.get('parsing_error')}. "
                                          f"Respond again with valid JSON only, in the required format.")]
        return None, raws

    def evaluate(self, direction: str, index: int, source: str, output: str, ground: str = "") -> ItemResult:
        module, src_lang, tgt_lang = DIRECTIONS[direction]
        ground = ground if direction == "fi-en" else ""
        system = module.system_prompt(self.glossary)
        res = ItemResult(
            id=item_id(source, output, ground), index=index, direction=direction,
            mode="reference_based" if ground.strip() else "reference_free",
            status="ok", source=source, output=output, ground=ground,
            model=self.model, base_url=self.base_url, prompt_version=module.PROMPT_VERSION,
            prompt_hash=prompt_hash(system), glossary=self.glossary,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        check = run_checks(source, output, src_lang, tgt_lang)
        if check.skip_status:
            res.status = check.skip_status
            return res

        t0 = time.perf_counter()
        samples: list[tuple[float, MQMAnnotation]] = []
        if not check.skip_llm:
            user = module.user_prompt(source=source, output=output, ground=ground)
            for _ in range(self.samples):
                ann, raws = self.annotate(system, user)
                res.raw += raws
                if ann is not None:
                    errs = merge_errors(check.errors, [JudgedError(**e.model_dump()) for e in ann.errors])
                    samples.append((mqm_penalty(errs), ann))
            if not samples:
                res.status = "judge_error"
                res.latency_s = time.perf_counter() - t0
                return res
        res.latency_s = time.perf_counter() - t0

        if samples:
            # keep the annotation whose penalty is closest to the median of all samples
            res.sample_penalties = [p for p, _ in samples]
            med = statistics.median(res.sample_penalties)
            _, ann = min(samples, key=lambda s: abs(s[0] - med))
            res.errors = merge_errors(check.errors, [JudgedError(**e.model_dump()) for e in ann.errors])
            res.overall_comment = ann.overall_comment
        else:
            res.errors = check.errors
            res.overall_comment = "Decided by rule-based checks only."

        res.penalty = mqm_penalty(res.errors)
        res.score = mqm_score(res.penalty, source)
        res.has_critical = any(e.severity == "critical" for e in res.errors)
        res.passed = is_pass(res.errors, res.penalty)
        if ground.strip():
            res.chrf = sentence_chrf(output, ground)
            res.bleu = sentence_bleu(output, ground)
        return res
