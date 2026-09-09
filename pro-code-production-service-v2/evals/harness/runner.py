"""Orchestrates an evaluation run against a live service."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import ServiceClient, ServiceError
from .dataset import dataset_fingerprint
from .judge import GeminiJudge
from .metrics import mean_and_stdev, percentile, score_behaviour, score_retrieval
from .types import CaseResult, GoldenCase

logger = logging.getLogger(__name__)


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    """
    Aggregate, keeping retrieval and generation strictly separate.

    Cases without ground truth are excluded from retrieval means rather than counted as
    zero, and the excluded count is reported so a shrinking denominator is visible.
    """
    ok = [r for r in results if r.status == "ok"]
    retrieval = [r.retrieval for r in ok if r.retrieval is not None]
    # Cases that declared ground truth but could not be scored: their pages are not in
    # the index. Reported separately so a shrinking denominator is never silent.
    unscoreable = [r.case_id for r in ok if r.retrieval is None and r.case_id.startswith("RA-")]
    judged = [r.judge for r in ok if r.judge is not None]
    behaviour = [r.behaviour for r in ok if r.behaviour is not None]

    summary: dict[str, Any] = {
        "cases_total": len(results),
        "cases_ok": len(ok),
        "cases_errored": len(results) - len(ok),
    }

    summary["retrieval"] = (
        {
            "cases_scored": len(retrieval),
            "cases_without_ground_truth": len(ok) - len(retrieval),
            "cases_unscoreable_missing_from_index": unscoreable,
            "mean_ground_truth_coverage": round(
                mean_and_stdev([r.coverage_ratio for r in retrieval])[0], 4
            ),
            "recall_at_k": round(mean_and_stdev([r.recall_at_k for r in retrieval])[0], 4),
            "recall_normalised": round(
                mean_and_stdev([r.recall_normalised for r in retrieval])[0], 4
            ),
            "mean_recall_ceiling": round(
                mean_and_stdev([r.recall_ceiling for r in retrieval])[0], 4
            ),
            "cases_missed_entirely": [i for i, r in enumerate(retrieval) if r.recall_at_k == 0.0],
            "precision_at_k": round(mean_and_stdev([r.precision_at_k for r in retrieval])[0], 4),
            "mrr": round(mean_and_stdev([r.mrr for r in retrieval])[0], 4),
            "ndcg_at_k": round(mean_and_stdev([r.ndcg_at_k for r in retrieval])[0], 4),
        }
        if retrieval
        else {"cases_scored": 0, "cases_without_ground_truth": len(ok)}
    )

    if judged:
        ctx = mean_and_stdev([j.context_relevance_mean for j in judged])
        grd = mean_and_stdev([j.groundedness_mean for j in judged])
        ans = mean_and_stdev([j.answer_relevance_mean for j in judged])
        summary["generation"] = {
            "cases_scored": len(judged),
            "judge_model": judged[0].judge_model,
            "trials_per_case": judged[0].trials,
            "context_relevance": round(ctx[0], 4),
            "groundedness": round(grd[0], 4),
            "answer_relevance": round(ans[0], 4),
            "triad": round((ctx[0] + grd[0] + ans[0]) / 3.0, 4),
            # Spread across cases, and the largest within-case spread across trials.
            # The latter is the judge's own instability, which Phase 0 measured at 0.06.
            "groundedness_spread_across_cases": round(grd[1], 4),
            "max_within_case_judge_stdev": round(
                max(
                    max(j.context_relevance_stdev, j.groundedness_stdev, j.answer_relevance_stdev)
                    for j in judged
                ),
                4,
            ),
        }
    else:
        summary["generation"] = {"cases_scored": 0, "judge_model": ""}

    if behaviour:
        summary["behaviour"] = {
            "expectation_met_rate": round(
                sum(1 for b in behaviour if b.expectation_met) / len(behaviour), 4
            ),
            "forbidden_content_hits": sum(len(b.forbidden_hits) for b in behaviour),
            "structure_completeness": round(
                mean_and_stdev([b.structure_completeness for b in behaviour])[0], 4
            ),
            # How many cases could support a groundedness measurement at all. This was
            # 0 of 5 for v1's fixture, which is why its groundedness was inconclusive.
            "cases_with_measurable_groundedness": sum(
                1 for b in behaviour if b.context_covers_answer
            ),
        }

    latencies = [r.latency_ms for r in ok]
    summary["operational"] = {
        "latency_p50_ms": round(percentile(latencies, 0.50), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "cost_usd_total": round(sum(r.cost_usd for r in ok), 8),
        "cost_usd_per_case": round(sum(r.cost_usd for r in ok) / len(ok), 8) if ok else 0.0,
    }
    return summary


async def run_evaluation(
    cases: list[GoldenCase],
    client: ServiceClient,
    *,
    dataset_paths: list[Path],
    judge: GeminiJudge | None = None,
    top_k: int = 5,
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """Run every case against the service, scoring what is scoreable."""
    results: list[CaseResult] = []

    # Establish the retrieval precondition before scoring anything. A recall figure is
    # only meaningful over a corpus that contains the ground truth, so cases whose
    # pages were never ingested are excluded rather than scored zero.
    try:
        coverage = await client.corpus_coverage()
    except ServiceError as exc:
        logger.warning("corpus coverage unavailable (%s); retrieval scored without it", exc)
        coverage = {}
    indexed: set[int] | None = (
        {page for pages in coverage.values() for page in pages} if coverage else None
    )
    if indexed:
        logger.info("corpus reports %d indexed page(s)", len(indexed))

    for case in cases:
        result = CaseResult(
            case_id=case.case_id, query=case.query, expectation=case.expectation, tags=case.tags
        )
        try:
            outcome = (
                await client.retrieve_only(case.query, top_k=top_k)
                if retrieval_only
                else await client.run_case(case.query, top_k=top_k)
            )
        except ServiceError as exc:
            # A service failure is recorded, not fatal: one broken case should not
            # discard the evidence from the rest of the set.
            result.status = "service_error"
            result.error = str(exc)[:300]
            results.append(result)
            logger.error("%s: service error: %s", case.case_id, exc)
            continue

        result.latency_ms = round(outcome.latency_ms, 1)
        result.cost_usd = outcome.cost_usd
        result.citations = outcome.citations
        result.answer = outcome.answer
        # Behaviour needs an answer to judge, so it is skipped in retrieval-only mode
        # rather than scored against an empty string.
        if not retrieval_only:
            result.behaviour = score_behaviour(case, outcome.answer, outcome.retrieved_chars)
        result.retrieval = score_retrieval(case, outcome.citations, indexed_pages=indexed)

        # A JudgeError deliberately propagates. It invalidates the whole generation
        # aggregate, and substituting a score for a failed judge is the v1 defect this
        # harness exists to avoid.
        if judge is not None:
            result.judge = judge.score(case.query, outcome.context, outcome.answer)
        results.append(result)
        if result.behaviour is not None:
            logger.info(
                "%s: %s expectation_met=%s latency=%.0fms",
                case.case_id,
                "refused" if result.behaviour.refused else "answered",
                result.behaviour.expectation_met,
                result.latency_ms,
            )
        else:
            pages = result.retrieval.retrieved_pages if result.retrieval else []
            logger.info("%s: retrieved %s latency=%.0fms", case.case_id, pages, result.latency_ms)

    return {
        "provenance": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "target": client.base_url,
            "evaluation_mode": "retrieval-only" if retrieval_only else "live-service",
            "datasets": [p.name for p in sorted(dataset_paths)],
            "dataset_fingerprint": dataset_fingerprint(dataset_paths),
            "judge_model": judge.model if judge else "",
            "judge_trials": judge.trials if judge else 0,
            "top_k": top_k,
            "indexed_pages": len(indexed) if indexed else 0,
        },
        "summary": _aggregate(results),
        "results": [r.model_dump(exclude={"answer"}) for r in results],
    }
