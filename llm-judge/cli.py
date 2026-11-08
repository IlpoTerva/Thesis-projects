"""
LLM-as-a-judge for the multilingual assistant's translations.

    python cli.py fi-en                # judge translate_in.json (queries, FI -> EN)
    python cli.py en-fi                # judge translate_out.json (answers, EN -> FI)
    python cli.py all --limit 5 --model gpt-4o --base-url https://api.openai.com/v1
"""
import argparse
import csv
import json
import os
import re
import sys

from config import settings
from judge import DIRECTIONS, Judge, item_id
from metrics import aggregate
from schemas import ItemResult


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _load_items(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            and r.glossary == judge.glossary
            # rule-only and skipped items have no LLM samples
            and (not r.sample_penalties or len(r.sample_penalties) == judge.samples))


def _write_csv(path: str, results: list[ItemResult]):
    cols = ["index", "id", "direction", "mode", "status", "penalty", "score", "passed", "has_critical",
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
        print(f"        - {e.severity:<8} {e.category:<24} [{e.origin}] '{e.span or e.source_span}' :: {e.explanation}")


def run_direction(direction: str, judge: Judge, input_path: str, limit: int | None, force: bool):
    module = DIRECTIONS[direction][0]
    items = _load_items(input_path)[:limit] if limit else _load_items(input_path)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    base = os.path.join(settings.RESULTS_DIR, f"{direction}__{_slug(judge.model)}")
    jsonl_path = base + ".jsonl"
    cache = _load_results(jsonl_path)

    print(f"\n=== {direction}  model={judge.model}  items={len(items)}  input={input_path}")
    current: list[ItemResult] = []
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for i, it in enumerate(items):
            source, output = it.get("In", ""), it.get("Out", "")
            ground = it.get("Ground", "") if direction == "fi-en" else ""
            cached = cache.get(item_id(source, output, ground))
            if not force and _is_cached(cached, judge, module.PROMPT_VERSION):
                r = cached.model_copy(update={"index": i})
            else:
                r = judge.evaluate(direction, i, source, output, ground)
                out.write(r.model_dump_json() + "\n")
                out.flush()
            current.append(r)
            _print_item(r)

    summary = {"direction": direction, "model": judge.model, "base_url": judge.base_url,
               "prompt_version": module.PROMPT_VERSION, "glossary": judge.glossary,
               "mqm_weights": settings.MQM_WEIGHTS, "input": input_path, **aggregate(current)}
    with open(f"{base}__summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_csv(f"{base}.csv", current)

    a = summary["all"]
    print(f"--- {direction}: scored={a['n']}  status={summary['status_counts']}")
    if a["n"]:
        print(f"    penalty mean={a['penalty_mean']:.2f} median={a['penalty_median']:.1f}  "
              f"score={a['score_mean']:.1f}  pass={a['pass_rate']:.0%}  critical={a['critical_rate']:.0%}")
    rb = summary["reference_based"]
    if rb.get("n"):
        print(f"    reference-based n={rb['n']}  chrF={rb['chrf_corpus']:.1f}  BLEU={rb['bleu_corpus']:.1f}  "
              f"spearman(penalty, chrF)={rb['spearman_penalty_vs_chrf']}")
    print(f"    results: {base}.jsonl | .csv | __summary.json")


def main():
    p = argparse.ArgumentParser(description="LLM-as-a-judge for FI<->EN translation quality (MQM)")
    p.add_argument("direction", choices=["fi-en", "en-fi", "all"])
    p.add_argument("--in", dest="in_file", default=settings.TRANSLATE_IN_FILE, help="FI->EN input json")
    p.add_argument("--out", dest="out_file", default=settings.TRANSLATE_OUT_FILE, help="EN->FI input json")
    p.add_argument("--model", help="override JUDGE_MODEL")
    p.add_argument("--base-url", help="override JUDGE_BASE_URL")
    p.add_argument("--samples", type=int, help="judge each item N times, keep median (JUDGE_SAMPLES)")
    p.add_argument("--limit", type=int, help="only the first N items")
    p.add_argument("--no-glossary", action="store_true", help="leave the domain glossary out of the prompt")
    p.add_argument("--force", action="store_true", help="re-judge items that already have results")
    args = p.parse_args()

    judge = Judge(model=args.model, base_url=args.base_url, samples=args.samples,
                  glossary=False if args.no_glossary else None)
    inputs = {"fi-en": args.in_file, "en-fi": args.out_file}
    for direction in (["fi-en", "en-fi"] if args.direction == "all" else [args.direction]):
        run_direction(direction, judge, inputs[direction], args.limit, args.force)


if __name__ == "__main__":
    sys.exit(main())
