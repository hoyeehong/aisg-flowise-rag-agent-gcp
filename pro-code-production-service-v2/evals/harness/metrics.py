"""
Deterministic metrics: retrieval quality and behavioural checks.

Everything here is computable without a model, which matters for two reasons. It can
gate CI without an API key or quota, and it isolates retrieval regressions from
generation regressions -- the distinction v1's single aggregate could not express.
"""

from __future__ import annotations

import math
import re
import statistics

from .types import BehaviourScores, Expectation, GoldenCase, RetrievalScores

# Citations arrive as "imda_report.pdf, p.31".
_CITATION = re.compile(r"^(?P<source>.+?),\s*p\.(?P<page>\d+)$")

# Phrases with which a grounded system declares the corpus cannot support an answer.
# A heuristic, deliberately broad: for a *refusal* check, missing a genuine refusal
# (and marking a correct behaviour wrong) is the costlier error.
_REFUSAL_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdoes not (?:provide|contain|mention|include|discuss|address|cover)\b",
        r"\bdo not (?:provide|contain|mention|include|discuss|address|cover)\b",
        r"\bno (?:information|data|mention|reference|evidence|content|passages?)\b",
        r"\bnot (?:available|present|found|covered|discussed|addressed|supported)\b",
        r"\bcannot (?:be )?(?:answer|answered|determine[d]?|confirm(?:ed)?)\b",
        r"\bcannot (?:be )?(?:verif(?:y|ied)|provide[d]?|assess(?:ed)?|establish(?:ed)?)\b",
        r"\bunable to (?:answer|determine|confirm|verify|provide|establish|assess)\b",
        r"\b(?:could|can) not be (?:determined|verified|confirmed|established|found)\b",
        r"\b(?:insufficient|no relevant) (?:context|information|evidence)\b",
        r"\boutside the scope\b",
        r"\bnot (?:in|within) the (?:report|source|provided context)\b",
        r"\bthe (?:report|source|context) (?:is silent|makes no)\b",
    )
)

_REQUIRED_SECTIONS = (
    r"executive summary",
    r"key findings|regional analysis",
    r"strategic enablers|recommendations",
    r"conclusion|future outlook",
)


def parse_citations(citations: list[str]) -> list[int]:
    """Extract page numbers from citation strings, preserving rank order."""
    pages: list[int] = []
    for citation in citations:
        match = _CITATION.match(citation.strip())
        if match:
            pages.append(int(match.group("page")))
    return pages


def detects_refusal(text: str) -> bool:
    """Whether the response declares the corpus cannot support an answer."""
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def structure_completeness(text: str) -> float:
    lowered = text.lower()
    present = sum(1 for pattern in _REQUIRED_SECTIONS if re.search(pattern, lowered))
    return present / len(_REQUIRED_SECTIONS)


def score_behaviour(case: GoldenCase, answer: str, context_chars: int) -> BehaviourScores:
    """
    Deterministic behavioural scoring.

    For a REFUSE case, structure completeness is not required: a correct refusal is a
    short statement of absence, and demanding four report sections would penalise
    exactly the behaviour being asked for.
    """
    refused = detects_refusal(answer)
    lowered = answer.lower()
    forbidden = [needle for needle in case.must_not_contain if needle.lower() in lowered]

    if case.expectation is Expectation.REFUSE:
        expectation_met = refused and not forbidden
    else:
        expectation_met = (not refused) and not forbidden

    return BehaviourScores(
        structure_completeness=structure_completeness(answer),
        refused=refused,
        expectation_met=expectation_met,
        forbidden_hits=forbidden,
        context_chars=context_chars,
        answer_chars=len(answer),
    )


def score_retrieval(
    case: GoldenCase,
    citations: list[str],
    *,
    indexed_pages: set[int] | None = None,
) -> RetrievalScores | None:
    """
    Rank-aware retrieval metrics, or ``None`` when the case cannot be scored.

    ``None`` is returned in two situations, and both matter:

    * the case has no verifiable ground truth, or
    * none of its ground-truth pages are in the index.

    Returning zeros instead would report a dataset gap or an incomplete ingest as a
    retriever failure. That is precisely the error Phase 0 found in v1, where a
    groundedness score measured a truncated fixture rather than the system, and it is
    why ``indexed_pages`` is a precondition rather than an afterthought.
    """
    if not case.scores_retrieval:
        return None

    expected_all = set(case.expected_pages)
    if indexed_pages is None:
        expected = expected_all
    else:
        expected = expected_all & indexed_pages
        if not expected:
            # Nothing to find: the corpus does not contain this case's ground truth.
            return None

    retrieved = parse_citations(citations)
    k = len(retrieved)

    coverage_ratio = len(expected) / len(expected_all) if expected_all else 1.0
    ceiling = min(1.0, k / len(expected)) if expected and k else 1.0

    if k == 0:
        return RetrievalScores(
            recall_at_k=0.0,
            recall_ceiling=1.0,
            recall_normalised=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            k=0,
            retrieved_pages=[],
            expected_pages=sorted(expected_all),
            expected_pages_indexed=sorted(expected),
            coverage_ratio=round(coverage_ratio, 4),
        )

    hits = [page in expected for page in retrieved]
    relevant_found = sum(hits)

    recall = relevant_found / len(expected)
    precision = relevant_found / k
    mrr = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)

    # Binary-relevance nDCG. The ideal ranking puts every relevant page first, capped
    # at k -- so a case with more ground-truth pages than k is not penalised for the
    # pages that could not fit.
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(expected))))
    ndcg = dcg / ideal if ideal > 0 else 0.0

    return RetrievalScores(
        recall_at_k=round(recall, 4),
        recall_ceiling=round(ceiling, 4),
        recall_normalised=round(recall / ceiling if ceiling > 0 else 0.0, 4),
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
        k=k,
        retrieved_pages=retrieved,
        expected_pages=sorted(expected_all),
        expected_pages_indexed=sorted(expected),
        coverage_ratio=round(coverage_ratio, 4),
    )


def mean_and_stdev(values: list[float]) -> tuple[float, float]:
    """Mean with sample standard deviation; stdev is 0 for a single observation."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def percentile(values: list[float], fraction: float) -> float:
    """
    Nearest-rank percentile.

    Deliberately not interpolated: with the tens of cases a golden set holds,
    interpolation invents a latency no request actually had.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]
