"""
Evaluation CLI.

    uv run python -m evals.harness --target http://localhost:8000

Modes:
  (default)          score the live service; judge metrics require --judge-key
  --gate             additionally compare against a baseline and exit non-zero on
                     regression, so CI can require it
  --write-baseline   record this run as the new baseline (never implicit)

EXIT CODES
  0  passed
  1  a gate finding: below a floor, or a regression beyond tolerance
  2  the run is unusable: dataset error, service unreachable, judge failure, or a
     baseline that is not comparable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import ServiceClient
from .dataset import DatasetError, discover_datasets, load_datasets
from .gate import (
    EXIT_OK,
    EXIT_UNUSABLE,
    baseline_is_comparable,
    evaluate_gate,
    load_baseline,
)
from .judge import DEFAULT_JUDGE_MODEL, DEFAULT_TRIALS, GeminiJudge, JudgeError
from .runner import run_evaluation

DEFAULT_BASELINE = Path(__file__).parent.parent / "baselines" / "current.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _print_summary(report: dict[str, Any]) -> None:
    prov, summary = report["provenance"], report["summary"]
    print("=" * 68)
    print("  Evaluation summary")
    print("=" * 68)
    print(f" target    : {prov['target']}")
    print(f" mode      : {prov['evaluation_mode']}")
    print(f" datasets  : {', '.join(prov['datasets'])}")
    print(f" fingerprint: {prov['dataset_fingerprint'][:16]}...")
    print(f" corpus    : {prov.get('indexed_pages', 0)} indexed page(s)")
    print(
        f" cases     : {summary['cases_ok']}/{summary['cases_total']} ok"
        + (f", {summary['cases_errored']} errored" if summary["cases_errored"] else "")
    )

    r = summary["retrieval"]
    print(f"\n RETRIEVAL  (scored on {r['cases_scored']} case(s))")
    if r.get("cases_without_ground_truth"):
        print(f"   {r['cases_without_ground_truth']} case(s) carry no verifiable ground truth")
    if r.get("cases_unscoreable_missing_from_index"):
        missing = ", ".join(r["cases_unscoreable_missing_from_index"])
        print(f"   EXCLUDED (ground-truth pages not in the index): {missing}")
        print("   -> scoring these would report an ingest gap as a retriever failure")
    if r["cases_scored"]:
        print(
            f"   recall@k {r['recall_at_k']:.3f} (ceiling {r.get('mean_recall_ceiling', 1.0):.3f})"
            f"   normalised {r.get('recall_normalised', 0.0):.3f}"
        )
        print(
            f"   precision@k {r['precision_at_k']:.3f}   MRR {r['mrr']:.3f}"
            f"   nDCG@k {r['ndcg_at_k']:.3f}"
        )
        print(
            f"   mean ground-truth coverage of the index: "
            f"{r.get('mean_ground_truth_coverage', 1.0):.1%}"
        )

    g = summary["generation"]
    print(f"\n GENERATION (scored on {g['cases_scored']} case(s))")
    if g["cases_scored"]:
        print(f"   judge {g['judge_model']} x{g['trials_per_case']} trial(s)")
        print(
            f"   context {g['context_relevance']:.3f}   groundedness {g['groundedness']:.3f}"
            f"   answer {g['answer_relevance']:.3f}   triad {g['triad']:.3f}"
        )
        print(
            f"   spread across cases {g['groundedness_spread_across_cases']:.3f}"
            f" | worst within-case judge stdev {g['max_within_case_judge_stdev']:.3f}"
        )
    else:
        print("   not scored (no judge key supplied)")

    if "behaviour" in summary:
        b = summary["behaviour"]
        print("\n BEHAVIOUR")
        print(
            f"   expectation met {b['expectation_met_rate']:.1%}"
            f" | forbidden hits {b['forbidden_content_hits']}"
            f" | structure {b['structure_completeness']:.3f}"
        )
        print(
            f"   cases where groundedness is measurable at all: "
            f"{b['cases_with_measurable_groundedness']}/{summary['cases_ok']}"
        )

    o = summary["operational"]
    print("\n OPERATIONAL")
    print(
        f"   latency p50 {o['latency_p50_ms']:.0f}ms  p95 {o['latency_p95_ms']:.0f}ms"
        f"  cost ${o['cost_usd_total']:.6f} (${o['cost_usd_per_case']:.6f}/case)"
    )
    print("=" * 68)


async def _run(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.dataset] if args.dataset else discover_datasets()
    try:
        cases = load_datasets(paths)
    except DatasetError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return EXIT_UNUSABLE
    print(f"loaded {len(cases)} case(s) from {len(paths)} dataset(s)")

    client = ServiceClient(args.target, timeout=args.timeout)
    code, body = await client.readyz()
    if code != 200:
        print(
            f"[FATAL] {args.target} is not ready (HTTP {code}): {json.dumps(body)[:200]}\n"
            f"        Evaluating a degraded service would measure the deployment, "
            f"not the system.",
            file=sys.stderr,
        )
        return EXIT_UNUSABLE

    judge = None
    if args.retrieval_only:
        print("[INFO] --retrieval-only: no model is invoked; scoring retrieval only")
    elif args.no_judge:
        print("[INFO] --no-judge: scoring retrieval and behaviour only")
    elif args.judge_key:
        judge = GeminiJudge(args.judge_key, model=args.judge_model, trials=args.trials)
    elif args.require_judge:
        print("[FATAL] --require-judge was set but no judge key was supplied.", file=sys.stderr)
        return EXIT_UNUSABLE
    else:
        print("[WARN] no judge key: generation metrics will not be scored")

    try:
        report = await run_evaluation(
            cases,
            client,
            dataset_paths=paths,
            judge=judge,
            top_k=args.top_k,
            retrieval_only=args.retrieval_only,
        )
    except JudgeError as exc:
        print(
            f"[FATAL] judge failed: {exc}\n        Refusing to substitute a score.", file=sys.stderr
        )
        return EXIT_UNUSABLE

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Timestamped, never a fixed path. The v1 runner wrote to a fixed filename, so any
    # run silently overwrote the published baseline -- which happened during Phase 2.
    out = Path(args.out) if args.out else REPORTS_DIR / f"eval-{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"\nreport: {out}")

    exit_code = EXIT_OK
    if args.gate:
        baseline_path = Path(args.baseline)
        baseline = load_baseline(baseline_path)
        if baseline is not None:
            reason = baseline_is_comparable(report, baseline)
            if reason:
                print(f"\n[FATAL] baseline not comparable: {reason}", file=sys.stderr)
                return EXIT_UNUSABLE
        exit_code, findings, notes = evaluate_gate(report["summary"], baseline)
        print("\nGATE")
        for note in notes:
            print(f"  note: {note}")
        if findings:
            for f in findings:
                print(f"  FAIL {f.metric} [{f.kind}]: {f.detail}")
        else:
            print("  passed" + ("" if baseline else " (floors only, no baseline)"))

    if args.write_baseline:
        # Explicit only. A baseline that updates itself on every run can never detect a
        # regression, because it always agrees with the latest result.
        baseline_path = Path(args.baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nbaseline written: {baseline_path}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.harness",
        description="Evaluate the running agent service against the golden sets.",
    )
    parser.add_argument("--target", default=os.getenv("EVAL_TARGET", "http://localhost:8000"))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="golden set path; repeatable. Default: every evals/golden/*.jsonl",
    )
    parser.add_argument(
        "--judge-key", default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="pinned judge model; a chain would blend graders",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="judge calls per case, averaged with reported spread",
    )
    parser.add_argument(
        "--require-judge", action="store_true", help="fail if no judge key is available"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "skip generation metrics even when a key is present. Retrieval and behaviour "
            "are deterministic, so this mode needs no API quota and is what CI gates on."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "score retrieval without generating reports. Invokes no chat model, so with "
            "the deterministic embedder it needs no credentials and no quota -- this is "
            "the mode CI gates on."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", default=None, help="report path (default: timestamped)")
    parser.add_argument("--gate", action="store_true", help="apply floors and regression checks")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record this run as the baseline (explicit only)",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
