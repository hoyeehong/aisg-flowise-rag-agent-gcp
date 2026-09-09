"""Graph state and the typed review decision that drives the HITL gate."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

ReviewAction = Literal["approve", "revise"]


class ReviewDecision(BaseModel):
    """
    A human reviewer's verdict on a draft.

    ``revise`` requires feedback: a revision request with no instruction produces an
    unguided rewrite, which in v1 was a plausible way to burn iterations without
    improving anything.
    """

    model_config = {"frozen": True}

    action: ReviewAction
    feedback: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _feedback_required_for_revision(self) -> ReviewDecision:
        if self.action == "revise" and not self.feedback.strip():
            raise ValueError("feedback is required when action is 'revise'")
        return self


class LLMCall(TypedDict):
    """One model invocation, recorded for per-request cost and latency accounting."""

    node: str
    model: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    fell_back: bool


class AgentState(TypedDict, total=False):
    """
    State threaded through the graph.

    ``llm_calls`` uses an additive reducer so every node appends its own accounting
    without clobbering earlier entries.
    """

    question: str
    top_k: int
    max_revisions: int

    context: str
    citations: list[str]
    retrieved_chars: int
    findings: str

    draft: str
    feedback_history: Annotated[list[str], operator.add]
    revisions: int

    decision: str
    final_report: str
    halted_reason: str

    llm_calls: Annotated[list[LLMCall], operator.add]
