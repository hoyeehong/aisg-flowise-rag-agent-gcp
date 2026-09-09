"""Request and response contracts for the HTTP surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["awaiting_review", "completed", "halted"]


class CreateReportRequest(BaseModel):
    """Start a new report run."""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    max_revisions: int | None = Field(default=None, ge=0, le=20)
    include_context: bool = Field(
        default=False,
        description=(
            "Return the retrieved context alongside the draft. Off by default because "
            "it is large and echoes corpus text. Required by the evaluation harness: "
            "groundedness and context relevance cannot be scored without the context "
            "the model actually saw."
        ),
    )


class ReviewRequest(BaseModel):
    """Submit a human verdict on a paused run."""

    model_config = {"extra": "forbid"}

    action: Literal["approve", "revise"]
    feedback: str = Field(default="", max_length=4000)


class CostSummary(BaseModel):
    """Per-run accounting, so cost is attributable at request granularity."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    models_used: list[str] = []
    fell_back: bool = False


class ReportState(BaseModel):
    """The externally visible state of a run."""

    thread_id: str
    status: RunStatus
    question: str
    draft: str = ""
    final_report: str = ""
    citations: list[str] = []
    retrieved_chars: int = 0
    context: str = Field(
        default="",
        description="Retrieved context; populated only when include_context was requested.",
    )
    revisions_used: int = 0
    revisions_allowed: int = 0
    feedback_history: list[str] = []
    halted_reason: str = ""
    prompt_versions: dict[str, str] = {}
    cost: CostSummary = CostSummary()


class RetrieveRequest(BaseModel):
    """Run retrieval alone, without invoking any model for generation."""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrieveResponse(BaseModel):
    """
    Retrieval output, with no generation.

    Exists so retrieval quality is measurable without spending model tokens. That makes
    recall, MRR and nDCG gateable in CI on every pull request -- with the deterministic
    embedder they need no API key and no quota at all, which the full report path cannot
    offer because it calls a chat model twice per case.
    """

    query: str
    citations: list[str] = []
    context: str = ""
    chunks: int = 0
    retrieved_chars: int = 0
    retriever: str = ""


class CorpusCoverage(BaseModel):
    """
    What is indexed for the active tenant.

    Exists so an evaluator can establish a precondition before scoring retrieval: a
    recall figure is only meaningful if the corpus actually contains the ground-truth
    pages it is measured against.
    """

    tenant_id: str
    sources: dict[str, list[int]] = {}
    total_pages: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total_pages == 0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, bool]
    detail: str = ""


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
