"""
Application service: translates between HTTP concerns and the agent graph.

Kept separate from the router so the orchestration logic is testable without a
transport, and so a future gRPC surface can reuse it unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .. import prompts
from ..agents import ReviewDecision
from ..agents.graph import AgentGraph
from ..agents.state import ReviewAction
from .schemas import CostSummary, ReportState, RunStatus


class RunNotFoundError(LookupError):
    """No checkpoint exists for the given thread id."""


class RunNotAwaitingReviewError(RuntimeError):
    """A review was submitted for a run that is not paused at the review gate."""


def _cost_summary(calls: list[dict[str, Any]]) -> CostSummary:
    return CostSummary(
        llm_calls=len(calls),
        prompt_tokens=sum(c["prompt_tokens"] for c in calls),
        completion_tokens=sum(c["completion_tokens"] for c in calls),
        total_cost_usd=round(sum(c["cost_usd"] for c in calls), 8),
        models_used=sorted({c["model"] for c in calls}),
        fell_back=any(c["fell_back"] for c in calls),
    )


class ReportService:
    """Starts, inspects and advances report runs."""

    def __init__(self, graph: AgentGraph, *, default_top_k: int, max_revisions: int) -> None:
        self._graph = graph
        self._default_top_k = default_top_k
        self._max_revisions = max_revisions

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _project(
        self,
        thread_id: str,
        values: dict[str, Any],
        *,
        paused: bool,
        include_context: bool = False,
    ) -> ReportState:
        """Build the external view of a run from raw graph state."""
        status: RunStatus
        if paused:
            status = "awaiting_review"
        elif values.get("halted_reason"):
            status = "halted"
        else:
            status = "completed"

        return ReportState(
            thread_id=thread_id,
            status=status,
            question=values.get("question", ""),
            draft=values.get("draft", ""),
            final_report=values.get("final_report", ""),
            citations=values.get("citations", []),
            retrieved_chars=values.get("retrieved_chars", 0),
            context=values.get("context", "") if include_context else "",
            revisions_used=values.get("revisions", 0),
            revisions_allowed=values.get("max_revisions", self._max_revisions),
            feedback_history=values.get("feedback_history", []),
            halted_reason=values.get("halted_reason", ""),
            prompt_versions=prompts.pinned_versions(),
            cost=_cost_summary(values.get("llm_calls", [])),
        )

    async def _is_paused(self, thread_id: str) -> bool:
        snapshot = await self._graph.aget_state(self._config(thread_id))
        return bool(snapshot.next)

    async def start(
        self,
        thread_id: str,
        question: str,
        *,
        tenant_id: str,
        top_k: int | None = None,
        max_revisions: int | None = None,
        include_context: bool = False,
    ) -> ReportState:
        """Run until the review gate (or to completion, if no gate is reached)."""
        await self._graph.ainvoke(
            {
                "question": question,
                "tenant_id": tenant_id,
                "top_k": top_k or self._default_top_k,
                "max_revisions": self._max_revisions if max_revisions is None else max_revisions,
            },
            config=self._config(thread_id),
        )
        return await self.get(thread_id, tenant_id=tenant_id, include_context=include_context)

    async def review(
        self, thread_id: str, action: ReviewAction, feedback: str, *, tenant_id: str
    ) -> ReportState:
        """Resume a paused run with a human verdict."""
        snapshot = await self._graph.aget_state(self._config(thread_id))
        if not snapshot.created_at or not self._owns(snapshot.values, tenant_id):
            raise RunNotFoundError(thread_id)
        if not snapshot.next:
            raise RunNotAwaitingReviewError(
                f"run {thread_id} is not awaiting review; it has already finished"
            )

        # Validate before resuming: a malformed decision should be a 422, not an
        # exception surfacing from inside the graph.
        decision = ReviewDecision(action=action, feedback=feedback)
        await self._graph.ainvoke(
            Command(resume=decision.model_dump()), config=self._config(thread_id)
        )
        return await self.get(thread_id, tenant_id=tenant_id)

    @staticmethod
    def _owns(values: dict[str, Any], tenant_id: str) -> bool:
        """
        Whether ``tenant_id`` may see this run.

        Application-level, not database-enforced: the LangGraph checkpoint tables
        have no tenant column, so the RLS policy protecting `chunks` does not cover
        conversation state. A run recorded before this check existed has no
        ``tenant_id`` and is treated as belonging to whoever asks, because failing
        closed on historical rows would make already-stored reports unreachable.
        Both facts are limitations, and both are documented in the README rather
        than implied to be solved.
        """
        recorded = values.get("tenant_id", "")
        return not recorded or recorded == tenant_id

    async def get(
        self, thread_id: str, *, tenant_id: str, include_context: bool = False
    ) -> ReportState:
        """
        Current state of a run.

        Async, and using ``aget_state`` rather than ``get_state``, because
        AsyncPostgresSaver rejects synchronous calls from the main thread with
        InvalidStateError. The in-memory saver tolerates the sync call, so a suite that
        only ever exercises InMemorySaver cannot catch this -- which is exactly how it
        reached a release.
        """
        snapshot = await self._graph.aget_state(self._config(thread_id))
        # 404 rather than 403 on a tenant mismatch: a caller who may not see a run must
        # not learn that it exists.
        if not snapshot.created_at or not self._owns(snapshot.values, tenant_id):
            raise RunNotFoundError(thread_id)
        return self._project(
            thread_id,
            snapshot.values,
            paused=bool(snapshot.next),
            include_context=include_context,
        )
