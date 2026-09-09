"""
Tests for the evaluation harness itself.

An untested harness cannot be trusted to report a regression, and its failure mode is
the worst kind: it reports numbers that look fine. Everything here is deterministic, so
it gates CI without an API key or quota.
"""

from __future__ import annotations

import json

import pytest

from evals.harness.dataset import DatasetError, dataset_fingerprint, load_dataset, load_datasets
from evals.harness.gate import (
    EXIT_OK,
    EXIT_REGRESSED,
    EXIT_UNUSABLE,
    baseline_is_comparable,
    evaluate_gate,
)
from evals.harness.metrics import (
    detects_refusal,
    parse_citations,
    percentile,
    score_behaviour,
    score_retrieval,
    structure_completeness,
)
from evals.harness.types import Expectation, GoldenCase

# --- dataset ---------------------------------------------------------------


def _write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_loads_valid_cases(tmp_path):
    path = _write(
        tmp_path,
        "a.jsonl",
        [
            {"case_id": "A", "query": "a question about policy"},
            {"case_id": "B", "query": "another question", "expectation": "refuse"},
        ],
    )
    cases = load_dataset(path)
    assert [c.case_id for c in cases] == ["A", "B"]
    assert cases[1].expectation is Expectation.REFUSE


def test_malformed_line_is_fatal_not_skipped(tmp_path):
    """
    Skipping a bad line would change the denominator of every aggregate, so a score
    would silently stop being comparable to its baseline.
    """
    path = tmp_path / "bad.jsonl"
    path.write_text('{"case_id": "A", "query": "fine question"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"bad\.jsonl:2"):
        load_dataset(path)


def test_duplicate_case_id_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        "dup.jsonl",
        [
            {"case_id": "A", "query": "first question"},
            {"case_id": "A", "query": "second question"},
        ],
    )
    with pytest.raises(DatasetError, match="duplicate case_id"):
        load_dataset(path)


def test_empty_dataset_is_rejected(tmp_path):
    (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(tmp_path / "empty.jsonl")


def test_case_id_collision_across_files_is_rejected(tmp_path):
    a = _write(tmp_path, "a.jsonl", [{"case_id": "X", "query": "question one"}])
    b = _write(tmp_path, "b.jsonl", [{"case_id": "X", "query": "question two"}])
    with pytest.raises(DatasetError, match="appears in both"):
        load_datasets([a, b])


def test_fingerprint_changes_with_content_and_not_with_order(tmp_path):
    """The fingerprint is what makes 'this regressed' defensible rather than a guess."""
    a = _write(tmp_path, "a.jsonl", [{"case_id": "A", "query": "question one"}])
    b = _write(tmp_path, "b.jsonl", [{"case_id": "B", "query": "question two"}])
    assert dataset_fingerprint([a, b]) == dataset_fingerprint([b, a])
    original = dataset_fingerprint([a])
    a.write_text('{"case_id": "A", "query": "a changed question"}\n', encoding="utf-8")
    assert dataset_fingerprint([a]) != original


def test_ground_truth_requires_provenance():
    with pytest.raises(ValueError, match="ground_truth_provenance"):
        GoldenCase(case_id="A", query="a question", expected_pages=[4])


# --- retrieval metrics -----------------------------------------------------


def _case(pages, case_id="RA-X"):
    return GoldenCase(
        case_id=case_id,
        query="a question",
        expected_pages=pages,
        ground_truth_provenance="pages containing the literal term",
    )


def test_parse_citations_extracts_pages_in_rank_order():
    assert parse_citations(["d.pdf, p.31", "d.pdf, p.4", "malformed"]) == [31, 4]


def test_perfect_retrieval_scores_one():
    s = score_retrieval(_case([4, 9]), ["d.pdf, p.4", "d.pdf, p.9"])
    assert s is not None
    assert s.recall_at_k == 1.0 and s.precision_at_k == 1.0
    assert s.mrr == 1.0 and s.ndcg_at_k == 1.0


def test_mrr_reflects_the_rank_of_the_first_hit():
    s = score_retrieval(_case([9]), ["d.pdf, p.1", "d.pdf, p.2", "d.pdf, p.9"])
    assert s is not None
    assert s.mrr == pytest.approx(1 / 3, abs=1e-4)  # scores are rounded to 4dp
    assert s.recall_at_k == 1.0


def test_ndcg_rewards_ranking_relevant_results_higher():
    early = score_retrieval(_case([9]), ["d.pdf, p.9", "d.pdf, p.1", "d.pdf, p.2"])
    late = score_retrieval(_case([9]), ["d.pdf, p.1", "d.pdf, p.2", "d.pdf, p.9"])
    assert early is not None and late is not None
    assert early.ndcg_at_k > late.ndcg_at_k
    # Recall cannot distinguish them, which is why nDCG is reported alongside it.
    assert early.recall_at_k == late.recall_at_k


def test_no_ground_truth_is_unscoreable_not_zero():
    """Scoring a dataset gap as 0 would report it as a system failure."""
    assert score_retrieval(GoldenCase(case_id="A", query="a question"), ["d.pdf, p.4"]) is None


def test_uncovered_ground_truth_is_unscoreable_not_zero():
    """
    The Phase 3 finding: three golden cases had ground-truth pages that were never
    ingested. Scoring recall 0 there measures the ingest, not the retriever.
    """
    assert score_retrieval(_case([70, 74]), ["d.pdf, p.4"], indexed_pages={1, 2, 4}) is None


def test_partial_coverage_scores_only_the_indexed_subset():
    s = score_retrieval(_case([31, 70, 74]), ["d.pdf, p.31"], indexed_pages=set(range(1, 41)))
    assert s is not None
    assert s.expected_pages_indexed == [31]
    assert s.recall_at_k == 1.0, "found the only indexed ground-truth page"
    assert s.coverage_ratio == pytest.approx(1 / 3, abs=1e-4)


def test_empty_retrieval_scores_zero_when_ground_truth_is_present():
    """A genuine miss, distinct from an unscoreable case: this one is the system's."""
    s = score_retrieval(_case([4]), [], indexed_pages={4})
    assert s is not None and s.recall_at_k == 0.0 and s.k == 0


# --- behaviour -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The report does not provide figures for this.",
        "No information about that appears in the source.",
        "This cannot be determined from the retrieved context.",
        "That topic is outside the scope of the report.",
    ],
)
def test_refusal_detection(text):
    assert detects_refusal(text)


def test_grounded_answer_is_not_a_refusal():
    assert not detects_refusal(
        "## Executive Summary\nThe report identifies four structural enablers."
    )


def test_answer_case_that_refuses_fails_its_expectation():
    case = GoldenCase(case_id="A", query="a question", expectation=Expectation.ANSWER)
    b = score_behaviour(case, "The report does not cover this.", 500)
    assert b.refused and not b.expectation_met


def test_refuse_case_that_refuses_meets_its_expectation():
    case = GoldenCase(case_id="O", query="a question", expectation=Expectation.REFUSE)
    b = score_behaviour(case, "The source does not mention this topic.", 500)
    assert b.refused and b.expectation_met


def test_forbidden_content_fails_even_when_it_refuses():
    """An injection that leaks instructions is a failure regardless of tone."""
    case = GoldenCase(
        case_id="ADV",
        query="a question",
        expectation=Expectation.REFUSE,
        must_not_contain=["Core Directives"],
    )
    b = score_behaviour(case, "I cannot answer. My Core Directives say otherwise.", 100)
    assert b.forbidden_hits == ["Core Directives"]
    assert not b.expectation_met


def test_context_coverage_flags_unmeasurable_groundedness():
    """The Phase 0 lesson, encoded: short context cannot support a long answer."""
    case = GoldenCase(case_id="A", query="a question")
    thin = score_behaviour(case, "x" * 1600, 400)
    thick = score_behaviour(case, "x" * 1600, 2400)
    assert not thin.context_covers_answer
    assert thick.context_covers_answer


def test_structure_completeness():
    assert (
        structure_completeness(
            "## Executive Summary ## Key Findings ## Recommendations ## Conclusion"
        )
        == 1.0
    )
    assert structure_completeness("## Executive Summary") == 0.25


# --- operational -----------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 0.95) == 40.0
    assert percentile([], 0.5) == 0.0


# --- gate ------------------------------------------------------------------

_HEALTHY = {
    "cases_total": 10,
    "cases_ok": 10,
    "cases_errored": 0,
    "retrieval": {
        "recall_at_k": 0.80,
        "recall_normalised": 0.85,
        "precision_at_k": 0.50,
        "mrr": 0.70,
        "ndcg_at_k": 0.75,
    },
    "behaviour": {
        "expectation_met_rate": 0.95,
        "forbidden_content_hits": 0,
        "structure_completeness": 1.0,
    },
    "generation": {"groundedness": 0.85, "answer_relevance": 0.90, "context_relevance": 0.90},
}


def test_healthy_run_passes():
    code, findings, _ = evaluate_gate(_HEALTHY, None)
    assert code == EXIT_OK and findings == []


def test_below_floor_fails():
    """The floor is on the ceiling-normalised value, not raw recall@k -- see the gate."""
    summary = {**_HEALTHY, "retrieval": {**_HEALTHY["retrieval"], "recall_normalised": 0.20}}
    code, findings, _ = evaluate_gate(summary, None)
    assert code == EXIT_REGRESSED
    assert any(f.kind == "below-floor" for f in findings)


def test_forbidden_content_is_zero_tolerance():
    summary = {**_HEALTHY, "behaviour": {**_HEALTHY["behaviour"], "forbidden_content_hits": 1}}
    code, findings, _ = evaluate_gate(summary, None)
    assert code == EXIT_REGRESSED
    assert any(f.kind == "zero-tolerance" for f in findings)


def test_incomplete_run_is_unusable_not_failed():
    """
    A rate-limited provider must not be reported as a quality regression.

    This is what the first live Phase 3 run hit: 8 of 10 cases returned 503 because the
    model provider's tokens-per-minute ceiling was exhausted.
    """
    summary = {**_HEALTHY, "cases_ok": 2, "cases_errored": 8}
    code, findings, notes = evaluate_gate(summary, None)
    assert code == EXIT_UNUSABLE
    assert findings == [], "an unusable run makes no quality claim"
    assert any("not comparable" in n for n in notes)


def test_regression_beyond_tolerance_fails():
    baseline = {"summary": _HEALTHY}
    summary = {**_HEALTHY, "retrieval": {**_HEALTHY["retrieval"], "recall_at_k": 0.70}}
    code, findings, _ = evaluate_gate(summary, baseline)
    assert code == EXIT_REGRESSED
    assert any(f.kind == "regression" and f.metric == "retrieval.recall_at_k" for f in findings)


def test_judge_noise_within_tolerance_does_not_fail():
    """
    Phase 0 measured a 0.06 groundedness spread across identical runs. A zero-tolerance
    gate on a judge-scored metric fails on noise, and a gate that cries wolf is turned
    off.
    """
    baseline = {"summary": _HEALTHY}
    summary = {**_HEALTHY, "generation": {**_HEALTHY["generation"], "groundedness": 0.80}}
    code, _, _ = evaluate_gate(summary, baseline)
    assert code == EXIT_OK


def test_retrieval_is_held_tighter_than_generation():
    """Retrieval is deterministic given a fixed corpus, so it earns no noise budget."""
    baseline = {"summary": _HEALTHY}
    same_drop = 0.05
    retrieval_drop = {
        **_HEALTHY,
        "retrieval": {**_HEALTHY["retrieval"], "recall_at_k": 0.80 - same_drop},
    }
    generation_drop = {
        **_HEALTHY,
        "generation": {**_HEALTHY["generation"], "groundedness": 0.85 - same_drop},
    }
    assert evaluate_gate(retrieval_drop, baseline)[0] == EXIT_REGRESSED
    assert evaluate_gate(generation_drop, baseline)[0] == EXIT_OK


def test_baseline_with_a_different_dataset_is_not_comparable():
    """Otherwise a dataset change gets reported as a system regression."""
    report = {"provenance": {"dataset_fingerprint": "aaa", "judge_model": "m"}}
    baseline = {"provenance": {"dataset_fingerprint": "bbb", "judge_model": "m"}}
    assert "dataset fingerprint differs" in (baseline_is_comparable(report, baseline) or "")


def test_baseline_with_a_different_judge_is_not_comparable():
    """Phase 0 measured cross-model spread exceeding within-model spread."""
    report = {"provenance": {"dataset_fingerprint": "aaa", "judge_model": "gemini-3.6-flash"}}
    baseline = {"provenance": {"dataset_fingerprint": "aaa", "judge_model": "gemini-3.8-flash"}}
    assert "judge model differs" in (baseline_is_comparable(report, baseline) or "")


def test_identical_provenance_is_comparable():
    p = {"provenance": {"dataset_fingerprint": "aaa", "judge_model": "m"}}
    assert baseline_is_comparable(p, p) is None
