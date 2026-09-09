"""
Regression gate: compare a run against a committed baseline.

Design notes, both learned the hard way in earlier phases:

* **Absolute floors and relative regressions are different questions.** A floor catches
  "this was never good enough"; a regression catches "this got worse". A gate with only
  floors passes a system that degrades from 0.95 to 0.81 against a 0.80 floor.
* **A tolerance is required, because the judge is not deterministic.** Phase 0 measured
  a 0.06 groundedness spread across identical runs. A zero-tolerance gate on a
  judge-scored metric would fail on noise, and a gate that cries wolf gets disabled.
  Deterministic retrieval metrics get a much tighter tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_DIR = Path(__file__).parent.parent / "baselines"

EXIT_OK = 0
EXIT_REGRESSED = 1
EXIT_UNUSABLE = 2

# Absolute floors. A metric below these is a failure regardless of the baseline.
FLOORS: dict[str, float] = {
    # The floor is on the ceiling-normalised value, because raw recall@k is bounded by
    # k/|expected| and would fail a healthy retriever on cases with many relevant pages.
    "retrieval.recall_normalised": 0.50,
    "retrieval.mrr": 0.40,
    "behaviour.expectation_met_rate": 0.80,
}

# Metrics that must never get worse than the baseline, with the tolerance each earns.
# Retrieval is deterministic given a fixed corpus and embedder, so it is held tightly.
# Generation is judge-scored, so its tolerance covers the measured judge spread.
TOLERANCES: dict[str, float] = {
    "retrieval.recall_at_k": 0.02,
    "retrieval.recall_normalised": 0.02,
    "retrieval.precision_at_k": 0.02,
    "retrieval.mrr": 0.02,
    "retrieval.ndcg_at_k": 0.02,
    "behaviour.expectation_met_rate": 0.05,
    "behaviour.structure_completeness": 0.05,
    "generation.groundedness": 0.08,
    "generation.answer_relevance": 0.08,
    "generation.context_relevance": 0.08,
}

# Any hit here is a hard failure: it means content was invented or instructions leaked.
ZERO_TOLERANCE = ("behaviour.forbidden_content_hits",)

# Fraction of cases that must complete for the run to mean anything. Below this the run
# is UNUSABLE rather than passed or failed: an aggregate over a fraction of the set is
# not comparable to a baseline over all of it, and a rate-limited provider would
# otherwise be reported as a quality result.
MIN_COMPLETION_RATE = 0.90


@dataclass
class Finding:
    metric: str
    kind: str
    current: float
    reference: float
    detail: str


def _lookup(summary: dict[str, Any], dotted: str) -> float | None:
    node: Any = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def evaluate_gate(
    summary: dict[str, Any], baseline: dict[str, Any] | None
) -> tuple[int, list[Finding], list[str]]:
    """Return ``(exit_code, findings, notes)``."""
    findings: list[Finding] = []
    notes: list[str] = []

    total = int(summary.get("cases_total", 0) or 0)
    ok = int(summary.get("cases_ok", 0) or 0)
    errored = int(summary.get("cases_errored", 0) or 0)
    completion = ok / total if total else 0.0

    if errored:
        notes.append(
            f"{errored} of {total} case(s) failed against the service; aggregates "
            f"cover only the {ok} that ran"
        )
    if total and completion < MIN_COMPLETION_RATE:
        # Deliberately not a regression finding. The run did not produce evidence, so
        # the honest verdict is "unusable", not "worse".
        notes.append(
            f"completion {completion:.0%} is below the {MIN_COMPLETION_RATE:.0%} "
            f"minimum: this run is not comparable to a baseline"
        )
        return EXIT_UNUSABLE, findings, notes

    for metric in ZERO_TOLERANCE:
        value = _lookup(summary, metric)
        if value:
            findings.append(
                Finding(
                    metric, "zero-tolerance", value, 0.0, "forbidden content appeared in a response"
                )
            )

    for metric, floor in FLOORS.items():
        value = _lookup(summary, metric)
        if value is None:
            notes.append(f"{metric}: not measured in this run, floor not applied")
            continue
        if value < floor:
            findings.append(
                Finding(metric, "below-floor", value, floor, f"{value:.4f} < floor {floor:.2f}")
            )

    if baseline is None:
        notes.append("no baseline: absolute floors applied, regressions not checked")
        return (EXIT_REGRESSED if findings else EXIT_OK), findings, notes

    base_summary = baseline.get("summary", {})
    for metric, tolerance in TOLERANCES.items():
        current = _lookup(summary, metric)
        reference = _lookup(base_summary, metric)
        if current is None or reference is None:
            continue
        if current < reference - tolerance:
            findings.append(
                Finding(
                    metric,
                    "regression",
                    current,
                    reference,
                    f"{current:.4f} vs baseline {reference:.4f} "
                    f"(drop {reference - current:.4f} > tolerance {tolerance:.2f})",
                )
            )

    return (EXIT_REGRESSED if findings else EXIT_OK), findings, notes


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def baseline_is_comparable(report: dict[str, Any], baseline: dict[str, Any]) -> str | None:
    """
    Whether a comparison is meaningful. Returns a reason when it is not.

    A score is only comparable to a baseline computed over the same cases with the same
    judge. Comparing across either is how a dataset change gets reported as a system
    regression.
    """
    current = report.get("provenance", {})
    reference = baseline.get("provenance", {})
    if current.get("dataset_fingerprint") != reference.get("dataset_fingerprint"):
        return (
            "dataset fingerprint differs from the baseline: the golden sets changed, so "
            "regression comparison would attribute a dataset change to the system"
        )
    if (
        current.get("judge_model")
        and reference.get("judge_model")
        and current["judge_model"] != reference["judge_model"]
    ):
        return (
            f"judge model differs (run {current['judge_model']} vs baseline "
            f"{reference['judge_model']}): cross-model spread exceeds within-model "
            f"spread, so generation metrics are not comparable"
        )
    return None
