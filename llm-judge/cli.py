"""
LLM-as-a-judge for the multilingual assistant's translations.

    python cli.py fi-en                # judge translate_in.json (queries, FI -> EN)
    python cli.py en-fi                # judge translate_out.json (answers, EN -> FI)
    python cli.py all --limit 5 --eval-model qwen7 --model gpt-4o --base-url https://api.openai.com/v1
    python cli.py compare              # combine per-model results into comparison files
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

from config import settings
from judge import DIRECTIONS, Judge, item_id
from metrics import aggregate
from schemas import ItemResult

# models whose translations are being evaluated (short name -> key / full model name)
EVAL_MODELS = {
    "qwen14": "qwen2.5:14b-instruct",
    "qwen7": "qwen2.5:7b-instruct",
    "llama3": "llama3.2:3b",
}

MISSING = "<missing>"  # error has a correct term but no span in the output (e.g. omission)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _load_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_items(path: str, eval_model: str, limit: int | None) -> list[dict]:
    """Items translated by one eval model. Accepts the short name or the full model name as the JSON key."""
    data = _load_items(path)
    if isinstance(data, dict):
        key = eval_model if eval_model in data else EVAL_MODELS.get(eval_model, eval_model)
        if key not in data:
            raise SystemExit(f"{path}: no items for eval model '{eval_model}' (keys: {', '.join(data)})")
        data = data[key]
    return data[:limit] if limit else data


def _load_results(path: str) -> dict[str, ItemResult]:
    """Latest result per item id."""
    results: dict[str, ItemResult] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = ItemResult.model_validate_json(line)
                    results[r.id] = r
    return results


def _is_cached(r: ItemResult | None, judge: Judge, prompt_version: str) -> bool:
    return (r is not None and r.status != "judge_error" and r.prompt_version == prompt_version
            # files are now named by eval model, so different judges share a file
            and r.model == judge.model
            and r.glossary == judge.glossary
            # rule-only and skipped items have no LLM samples
            and (not r.sample_penalties or len(r.sample_penalties) == judge.samples))


def _norm(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _term_errors(results: list[ItemResult], direction: str) -> dict[str, list[str]]:
    """{correct term: [wrong translations, most frequent first]}, terms with most errors first.

    Correct term = reference text (fi-en, `reference_span`) or English source text (en-fi, `source_span`).
    Errors without a correct term (e.g. pure fluency/grammar errors) are left out.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for e in r.errors:
            correct = getattr(e, "reference_span", "") if direction == "fi-en" else ""
            # fi-en items without a reference fall back to the Finnish source term
            correct = _norm(correct or e.source_span)
            if not correct:
                continue
            counts[correct][_norm(e.span) or MISSING] += 1
    ordered = sorted(counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    return {term: [w for w, _ in c.most_common()] for term, c in ordered}


def _write_csv(path: str, results: list[ItemResult]):
    cols = ["eval_model", "index", "id", "direction", "mode", "status", "penalty", "score", "passed", "has_critical",
            "n_errors", "n_minor", "n_major", "n_critical", "chrf", "bleu",
            "source", "output", "ground", "overall_comment"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            sev = [e.severity for e in r.errors]
            w.writerow({**r.model_dump(include=set(cols)),
                        "n_errors": len(sev), "n_minor": sev.count("minor"),
                        "n_major": sev.count("major"), "n_critical": sev.count("critical")})


def _fmt(v, nd=1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _print_item(r: ItemResult):
    src = r.source.replace("\n", " ")[:45]
    print(f"[{r.index:>3}] {r.mode:<15} {r.status:<26} pen={_fmt(r.penalty):>5} "
          f"crit={'Y' if r.has_critical else '-'} chrF={_fmt(r.chrf):>5}  {src}")
    for e in r.errors:
        ref = getattr(e, "reference_span", "")
        ref = f" (ref: '{ref}')" if ref else ""
        print(f"        - {e.severity:<8} {e.category:<24} [{e.origin}] '{e.span or e.source_span}'{ref} :: {e.explanation}")


def run_direction(direction: str, eval_model: str, judge: Judge, input_path: str, limit: int | None, force: bool):
    module = DIRECTIONS[direction][0]
    items = _load_model_items(input_path, eval_model, limit)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    base = os.path.join(settings.RESULTS_DIR, f"{direction}__{_slug(eval_model)}")
    jsonl_path = base + ".jsonl"
    cache = _load_results(jsonl_path)

    print(f"\n=== {direction}  eval_model={eval_model}  judge={judge.model}  items={len(items)}  input={input_path}")
    current: list[ItemResult] = []
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for i, it in enumerate(items):
            source, output = it.get("original", ""), it.get("translated", "")
            ground = it.get("Ground", "") if direction == "fi-en" else ""
            cached = cache.get(item_id(source, output, ground))
            if not force and _is_cached(cached, judge, module.PROMPT_VERSION):
                r = cached.model_copy(update={"index": i, "eval_model": eval_model})
            else:
                r = judge.evaluate(direction, i, source, output, ground)
                r.eval_model = eval_model
                out.write(r.model_dump_json() + "\n")
                out.flush()
            current.append(r)
            _print_item(r)

    summary = {"direction": direction, "eval_model": eval_model,
               "eval_model_name": EVAL_MODELS.get(eval_model, eval_model),
               "model": judge.model, "base_url": judge.base_url,
               "prompt_version": module.PROMPT_VERSION, "glossary": judge.glossary,
               "mqm_weights": settings.MQM_WEIGHTS, "input": input_path, **aggregate(current)}
    with open(f"{base}__summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_csv(f"{base}.csv", current)
    with open(f"{base}__terms.json", "w", encoding="utf-8") as f:
        json.dump(_term_errors(current, direction), f, ensure_ascii=False, indent=2)

    a = summary["all"]
    print(f"--- {direction}: scored={a['n']}  status={summary['status_counts']}")
    if a["n"]:
        print(f"    penalty mean={a['penalty_mean']:.2f} median={a['penalty_median']:.1f}  "
              f"score={a['score_mean']:.1f}  pass={a['pass_rate']:.0%}  critical={a['critical_rate']:.0%}")
    rb = summary["reference_based"]
    if rb.get("n"):
        print(f"    reference-based n={rb['n']}  chrF={rb['chrf_corpus']:.1f}  BLEU={rb['bleu_corpus']:.1f}  "
              f"spearman(penalty, chrF)={rb['spearman_penalty_vs_chrf']}")
    print(f"    results: {base}.jsonl | .csv | __summary.json | __terms.json")


COMPARE_COLS = ["eval_model", "eval_model_name", "judge_model", "prompt_version", "glossary",
                "n", "penalty_mean", "penalty_median", "score_mean", "pass_rate", "critical_rate",
                "ref_n", "chrf_corpus", "bleu_corpus", "spearman_penalty_vs_chrf", "status_counts"]


def compare_direction(direction: str):
    """Combine the per-eval-model summary and terms files of one direction."""
    rows, terms_by_model = [], defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(settings.RESULTS_DIR, f"{direction}__*__summary.json"))):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "eval_model" not in s:  # old file named after the judge model
            print(f"    skipping {path} (no eval_model, made before the rename)")
            continue
        a, rb = s.get("all", {}), s.get("reference_based", {})
        rows.append({"eval_model": s["eval_model"], "eval_model_name": s.get("eval_model_name", ""),
                     "judge_model": s.get("model", ""), "prompt_version": s.get("prompt_version", ""),
                     "glossary": s.get("glossary", ""), "n": a.get("n"),
                     **{k: a.get(k) for k in ("penalty_mean", "penalty_median", "score_mean",
                                              "pass_rate", "critical_rate")},
                     "ref_n": rb.get("n"), "chrf_corpus": rb.get("chrf_corpus"),
                     "bleu_corpus": rb.get("bleu_corpus"),
                     "spearman_penalty_vs_chrf": rb.get("spearman_penalty_vs_chrf"),
                     "status_counts": json.dumps(s.get("status_counts", {}), ensure_ascii=False)})
        terms_path = path.replace("__summary.json", "__terms.json")
        if os.path.exists(terms_path):
            with open(terms_path, "r", encoding="utf-8") as f:
                for term, wrong in json.load(f).items():
                    terms_by_model[term][s["eval_model"]] = wrong

    if not rows:
        print(f"--- {direction}: no per-model results to compare")
        return
    if len({(r["judge_model"], r["prompt_version"], r["glossary"]) for r in rows}) > 1:
        print(f"    WARNING: {direction} results come from different judges / prompt versions / glossary "
              f"settings, so they are not directly comparable")

    base = os.path.join(settings.RESULTS_DIR, f"{direction}__")
    with open(base + "comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARE_COLS)
        w.writeheader()
        w.writerows(rows)
    # terms that the most models get wrong first, then by total number of distinct wrong translations
    ordered = sorted(terms_by_model.items(),
                     key=lambda kv: (-len(kv[1]), -sum(len(v) for v in kv[1].values()), kv[0]))
    with open(base + "terms_by_model.json", "w", encoding="utf-8") as f:
        json.dump(dict(ordered), f, ensure_ascii=False, indent=2)

    print(f"\n=== {direction}: compared {len(rows)} eval models")
    for r in rows:
        print(f"    {r['eval_model']:<8} n={r['n']}  penalty={_fmt(r['penalty_mean'], 2)}  "
              f"pass={_fmt(None if r['pass_rate'] is None else 100 * r['pass_rate'], 0)}%  "
              f"chrF={_fmt(r['chrf_corpus'])}")
    print(f"    results: {base}comparison.csv | {base}terms_by_model.json")


def main():
    p = argparse.ArgumentParser(description="LLM-as-a-judge for FI<->EN translation quality (MQM)")
    p.add_argument("direction", choices=["fi-en", "en-fi", "all", "compare"],
                   help="direction to judge, or 'compare' to combine existing per-model results")
    p.add_argument("--in", dest="in_file", default=settings.TRANSLATE_IN_FILE, help="FI->EN input json")
    p.add_argument("--out", dest="out_file", default=settings.TRANSLATE_OUT_FILE, help="EN->FI input json")
    p.add_argument("--model", help="override JUDGE_MODEL")
    p.add_argument("--eval-model", default="qwen14", choices=list(EVAL_MODELS),
                   help="model whose translations are evaluated")
    p.add_argument("--base-url", help="override JUDGE_BASE_URL")
    p.add_argument("--samples", type=int, help="judge each item N times, keep median (JUDGE_SAMPLES)")
    p.add_argument("--limit", type=int, help="only the first N items")
    p.add_argument("--no-glossary", action="store_true", help="leave the domain glossary out of the prompt")
    p.add_argument("--force", action="store_true", help="re-judge items that already have results")
    args = p.parse_args()

    if args.direction == "compare":  # no judge calls, just reads existing results
        for direction in ("fi-en", "en-fi"):
            compare_direction(direction)
        return

    judge = Judge(model=args.model, base_url=args.base_url, samples=args.samples,
                  glossary=False if args.no_glossary else None)
    inputs = {"fi-en": args.in_file, "en-fi": args.out_file}
    for direction in (["fi-en", "en-fi"] if args.direction == "all" else [args.direction]):
        run_direction(direction, args.eval_model, judge, inputs[direction], args.limit, args.force)


if __name__ == "__main__":
    sys.exit(main())