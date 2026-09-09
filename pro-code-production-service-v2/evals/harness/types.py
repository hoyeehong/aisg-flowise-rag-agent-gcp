"""
Golden-case and result contracts.

Two design choices carried over from earlier phases:

* **Retrieval and generation are scored separately.** In v1 a retrieval regression and
  a generation regression were indistinguishable in the aggregate, so neither could be
  acted on.
* **Absent ground truth is skipped, never scored as zero.** A case with no verifiable
  ``expected_pages`` contributes generation metrics only. Scoring it 0 would let the
  dataset's gaps masquerade as the system's failures -- the same class of error as v1
  reporting groundedness against a truncated context.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Expectation(StrEnum):
    """What a correct response looks like for this case."""

    ANSWER = "answer"
    # The corpus cannot support an answer, so the correct behaviour is to say so.
    # Treating these as ANSWER cases would reward confident invention.
    REFUSE = "refuse"


class GoldenCase(BaseModel):
    """One evaluation case. Immutable once released; a change means a new dataset version."""

    model_config = {"frozen": True, "extra": "forbid"}

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=3, max_length=2000)
    expectation: Expectation = Expectation.ANSWER

    # Verifiable retrieval ground truth. Empty means retrieval is NOT scored for this
    # case -- see the module docstring.
    expected_pages: list[int] = []
    source: str = "imda_report.pdf"
    ground_truth_provenance: str = ""

    reference_answer: str = ""
    # Strings whose presence indicates the response invented content or leaked
    # instructions. Checked case-insensitively.
    must_not_contain: list[str] = []
    tags: list[str] = []

    @property
    def scores_retrieval(self) -> bool:
        return bool(self.expected_pages)

    @model_validator(mode="after")
    def _ground_truth_needs_provenance(self) -> GoldenCase:
        if self.expected_pages and not self.ground_truth_provenance.strip():
            raise ValueError(
                f"{self.case_id}: expected_pages requires ground_truth_provenance so a "
                f"reader can check how the ground truth was established"
            )
        return self


class RetrievalScores(BaseModel):
    """Rank-aware retrieval quality against ``expected_pages``."""

    model_config = {"frozen": True}

    recall_at_k: float
    # Recall is bounded by k/|expected|: with 19 relevant pages and k=5, raw recall
    # cannot exceed 0.26 however good the retriever is. Reporting only the raw figure
    # makes a structurally capped case look like a 62% miss, so the ceiling and the
    # ceiling-normalised value are recorded alongside it. The normalised value answers
    # "of what was retrievable at this k, how much did we retrieve", which is the
    # question a gate should be asking and is comparable across cases.
    recall_ceiling: float = 1.0
    recall_normalised: float = 0.0
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    retrieved_pages: list[int] = []
    # The ground-truth pages actually present in the index. Recall is computed against
    # this subset, not against expected_pages, so an un-ingested page cannot be
    # reported as a retriever miss.
    expected_pages: list[int] = []
    expected_pages_indexed: list[int] = []
    coverage_ratio: float = 1.0


class JudgeScores(BaseModel):
    """
    Generation quality, with spread.

    Phase 0 measured a 0.06 groundedness spread across four runs of one model at
    temperature 0, and a wider spread across models. A single trial is a sample, so the
    harness runs several and reports mean with standard deviation; a mean quoted without
    spread overstates its own precision.
    """

    model_config = {"frozen": True}

    context_relevance_mean: float
    groundedness_mean: float
    answer_relevance_mean: float
    context_relevance_stdev: float = 0.0
    groundedness_stdev: float = 0.0
    answer_relevance_stdev: float = 0.0
    trials: int = 1
    judge_model: str = ""
    reasoning: str = ""

    @property
    def triad_mean(self) -> float:
        return (
            self.context_relevance_mean + self.groundedness_mean + self.answer_relevance_mean
        ) / 3.0


class BehaviourScores(BaseModel):
    """Deterministic checks that need no judge."""

    model_config = {"frozen": True}

    structure_completeness: float
    refused: bool
    expectation_met: bool
    forbidden_hits: list[str] = []
    context_chars: int = 0
    answer_chars: int = 0

    @property
    def context_covers_answer(self) -> bool:
        """
        Whether groundedness is even measurable for this case.

        The Phase 0 lesson: with context far shorter than the answer, a low groundedness
        score measures the fixture, not the system.
        """
        return self.answer_chars > 0 and self.context_chars / self.answer_chars >= 0.5


class CaseResult(BaseModel):
    """Everything recorded for one case."""

    case_id: str
    query: str
    expectation: Expectation
    tags: list[str] = []
    status: str = "ok"
    error: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    citations: list[str] = []
    answer: str = ""
    behaviour: BehaviourScores | None = None
    retrieval: RetrievalScores | None = None
    judge: JudgeScores | None = None
