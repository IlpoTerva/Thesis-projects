"""MQM scoring, chrF/BLEU baselines and run-level aggregation."""
from __future__ import annotations
import statistics
from collections import Counter
from typing import Iterable

from sacrebleu.metrics import BLEU, CHRF

from config import settings
from schemas import ErrorSpan, ItemResult


_CHRF = CHRF()
_BLEU_SENT = BLEU(effective_order=True)
_BLEU_CORPUS = BLEU()


def mqm_penalty(errors: Iterable[ErrorSpan]) -> float:
    return sum(settings.MQM_WEIGHTS[e.severity] for e in errors)


def mqm_score(penalty: float, source: str) -> float:
    """Penalty normalised per 100 source words, mapped to 0..100 (100 = no errors)."""
    words = max(1, len(source.split()))
    return max(0.0, 100.0 - penalty * 100.0 / words)


def is_pass(errors: list[ErrorSpan], penalty: float, source: str) -> bool:
    if any(e.severity == "critical" for e in errors):
        return False
    words = max(settings.MIN_NORM_WORDS, len(source.split()))  # e.g. 20
    return penalty * 100.0 / words <= settings.PASS_MAX_PENALTY_PER_100


def sentence_chrf(hyp: str, ref: str) -> float:
    return _CHRF.sentence_score(hyp, [ref]).score


def sentence_bleu(hyp: str, ref: str) -> float:
    return _BLEU_SENT.sentence_score(hyp, [ref]).score


def _spearman(x: list[float], y: list[float]) -> dict | None:
    if len(x) < settings.MIN_ITEMS_FOR_CORRELATION or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}


def _group_stats(items: list[ItemResult]) -> dict:
    if not items:
        return {"n": 0}
    pens = [r.penalty for r in items]
    errs = [e for r in items for e in r.errors]
    by_cat = Counter(f"{e.category}|{e.severity}" for e in errs)
    return {
        "n": len(items),
        "penalty_mean": statistics.mean(pens),
        "penalty_median": statistics.median(pens),
        "score_mean": statistics.mean(r.score for r in items),
        "pass_rate": sum(r.passed for r in items) / len(items),
        "critical_rate": sum(r.has_critical for r in items) / len(items),
        "errors_total": len(errs),
        "errors_by_severity": dict(Counter(e.severity for e in errs)),
        "errors_by_category_severity": dict(sorted(by_cat.items())),
        "errors_by_origin": dict(Counter(e.origin for e in errs)),
    }


def aggregate(results: list[ItemResult]) -> dict:
    scored = [r for r in results if r.status == "ok"]
    summary = {
        "items_total": len(results),
        "status_counts": dict(Counter(r.status for r in results)),
        "all": _group_stats(scored),
        "reference_based": _group_stats([r for r in scored if r.mode == "reference_based"]),
        "reference_free": _group_stats([r for r in scored if r.mode == "reference_free"]),
    }

    ref = [r for r in scored if r.mode == "reference_based"]
    if ref:
        hyps, refs = [r.output for r in ref], [r.ground for r in ref]
        summary["reference_based"]["chrf_corpus"] = _CHRF.corpus_score(hyps, [refs]).score
        summary["reference_based"]["bleu_corpus"] = _BLEU_CORPUS.corpus_score(hyps, [refs]).score
        summary["reference_based"]["chrf_sentence_mean"] = statistics.mean(r.chrf for r in ref)
        # Higher penalty should mean lower chrF -> expect a negative rho if judge and metric agree
        summary["reference_based"]["spearman_penalty_vs_chrf"] = _spearman(
            [r.penalty for r in ref], [r.chrf for r in ref])
    return summary
