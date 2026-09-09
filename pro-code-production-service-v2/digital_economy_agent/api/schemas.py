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
    revisions_used: int = 0
    revisions_allowed: int = 0
    feedback_history: list[str] = []
    halted_reason: str = ""
    prompt_versions: dict[str, str] = {}
    cost: CostSummary = CostSummary()


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
