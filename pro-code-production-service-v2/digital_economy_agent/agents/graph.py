"""
The research → write → human review → revise graph.

This is a code-owned port of the v1 Flowise Agentflow. Two things change by moving it
into LangGraph:

* The human review gate becomes ``interrupt()`` against a checkpointer, so a paused run
  is durable state addressed by ``thread_id`` rather than an open UI session.
* The revision loop is bounded. v1's loop node had no ceiling; here ``max_revisions``
  halts the graph with ``halted_reason`` set, so a reviewer who keeps requesting changes
  cannot spend tokens indefinitely.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from .. import prompts
from ..gateway import ChatMessage, ModelGateway
from ..observability import metrics as obs_metrics
from ..observability import span
from ..tools import RetrievalRequest, Retriever
from .state import AgentState, LLMCall, ReviewDecision

DEFAULT_MAX_REVISIONS = 3

# The graph is dynamically typed at the LangGraph boundary; AgentState is enforced by
# the node signatures rather than by these parameters.
AgentGraph = CompiledStateGraph[Any, Any, Any, Any]


def _record(node: str, prompt_id: str, completion: object) -> LLMCall:
    """
    Flatten a gateway Completion into a state-recordable accounting entry, and meter it.

    Metering here rather than in the gateway keeps the ``node`` label available: the
    operational question is which *step* is spending, not merely which model.
    """
    c = completion  # typed loosely to keep this helper free of a circular import
    obs_metrics.record_llm_call(
        model=c.model,  # type: ignore[attr-defined]
        node=node,
        prompt_tokens=c.usage.prompt_tokens,  # type: ignore[attr-defined]
        completion_tokens=c.usage.completion_tokens,  # type: ignore[attr-defined]
        cost_usd=c.cost_usd,  # type: ignore[attr-defined]
        fell_back=c.fell_back,  # type: ignore[attr-defined]
    )
    return LLMCall(
        node=node,
        model=c.model,  # type: ignore[attr-defined]
        prompt_version=prompt_id,
        prompt_tokens=c.usage.prompt_tokens,  # type: ignore[attr-defined]
        completion_tokens=c.usage.completion_tokens,  # type: ignore[attr-defined]
        cost_usd=c.cost_usd,  # type: ignore[attr-defined]
        latency_ms=c.latency_ms,  # type: ignore[attr-defined]
        fell_back=c.fell_back,  # type: ignore[attr-defined]
    )


def build_graph(
    gateway: ModelGateway,
    retriever: Retriever,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> AgentGraph:
    """Compile the agent graph. A checkpointer is required for the HITL pause to survive."""

    async def research(state: AgentState) -> AgentState:
        """Retrieve context, then synthesise grounded findings from it."""
        question = state["question"]
        top_k = state.get("top_k", 5)
        with span("retrieval", top_k=top_k) as retrieval_span:
            result = await retriever.retrieve(RetrievalRequest(query=question, top_k=top_k))
            retrieval_span.set_attribute("chunks", len(result.chunks))
            retrieval_span.set_attribute("chars", result.total_chars)
        obs_metrics.RETRIEVAL_CHUNKS.observe(len(result.chunks))
        if not result.chunks:
            obs_metrics.RETRIEVAL_EMPTY.inc()

        # An empty retrieval is reported, not papered over. Writing a report from no
        # context is precisely the failure v1's groundedness metric could not detect.
        if not result.chunks:
            return AgentState(
                context="",
                citations=[],
                retrieved_chars=0,
                findings=(
                    "No relevant context was retrieved for this question, so no grounded "
                    "findings can be produced."
                ),
                llm_calls=[],
            )

        prompt_id = prompts.pinned_versions()["research"]
        template = prompts.load("research")
        completion = await gateway.complete(
            [
                ChatMessage(
                    role="user",
                    content=template.format(context=result.context, question=question),
                )
            ],
            temperature=0.3,
        )
        return AgentState(
            context=result.context,
            citations=[c.citation() for c in result.chunks],
            retrieved_chars=result.total_chars,
            findings=completion.text,
            llm_calls=[_record("research", prompt_id, completion)],
        )

    async def write_draft(state: AgentState) -> AgentState:
        """Turn findings into the four-section executive report."""
        prompt_id = prompts.pinned_versions()["write_draft"]
        template = prompts.load("write_draft")
        with span("write_draft", prompt_version=prompt_id):
            completion = await gateway.complete(
                [
                    ChatMessage(
                        role="user",
                        content=template.format(
                            findings=state.get("findings", ""), question=state["question"]
                        ),
                    )
                ],
                temperature=0.3,
            )
        return AgentState(
            draft=completion.text,
            llm_calls=[_record("write_draft", prompt_id, completion)],
        )

    def await_review(state: AgentState) -> AgentState:
        """
        Human-in-the-loop gate.

        ``interrupt`` suspends the graph and persists it via the checkpointer. The run
        resumes only when a caller supplies a ``Command(resume=...)`` for this thread,
        so nothing is generated past this point without a human verdict.
        """
        obs_metrics.REVIEW_PAUSES.inc()
        raw = interrupt(
            {
                "draft": state.get("draft", ""),
                "citations": state.get("citations", []),
                "revisions_used": state.get("revisions", 0),
                "revisions_allowed": state.get("max_revisions", DEFAULT_MAX_REVISIONS),
            }
        )
        decision = raw if isinstance(raw, ReviewDecision) else ReviewDecision.model_validate(raw)

        obs_metrics.REVIEW_DECISIONS.labels(action=decision.action).inc()
        if decision.action == "approve":
            return AgentState(decision="approve", final_report=state.get("draft", ""))
        return AgentState(decision="revise", feedback_history=[decision.feedback])

    async def revise(state: AgentState) -> AgentState:
        """Apply the reviewer's feedback to the current draft."""
        prompt_id = prompts.pinned_versions()["revise"]
        template = prompts.load("revise")
        history = state.get("feedback_history", [])
        completion = await gateway.complete(
            [
                ChatMessage(
                    role="user",
                    content=template.format(
                        findings=state.get("findings", ""),
                        draft=state.get("draft", ""),
                        feedback="\n".join(f"- {f}" for f in history),
                    ),
                )
            ],
            temperature=0.3,
        )
        return AgentState(
            draft=completion.text,
            revisions=state.get("revisions", 0) + 1,
            llm_calls=[_record("revise", prompt_id, completion)],
        )

    def route_after_review(state: AgentState) -> Literal["revise", "__end__"]:
        if state.get("decision") == "approve":
            return "__end__"
        return "revise"

    def route_after_revision(state: AgentState) -> Literal["await_review", "__end__"]:
        """Bound the loop: halt rather than accept unlimited revision requests."""
        used = state.get("revisions", 0)
        allowed = state.get("max_revisions", DEFAULT_MAX_REVISIONS)
        if used >= allowed:
            return "__end__"
        return "await_review"

    async def halt_check(state: AgentState) -> AgentState:
        """Record why the graph ended if it ended by exhausting the revision budget."""
        used = state.get("revisions", 0)
        allowed = state.get("max_revisions", DEFAULT_MAX_REVISIONS)
        if used >= allowed and state.get("decision") != "approve":
            obs_metrics.REVISION_BUDGET_EXHAUSTED.inc()
            return AgentState(
                final_report=state.get("draft", ""),
                halted_reason=(
                    f"revision budget exhausted after {used} of {allowed} allowed "
                    f"revisions; returning the latest draft without human approval"
                ),
            )
        return AgentState()

    builder = StateGraph(AgentState)
    builder.add_node("research", research)
    builder.add_node("write_draft", write_draft)
    builder.add_node("await_review", await_review)
    builder.add_node("revise", revise)
    builder.add_node("halt_check", halt_check)

    builder.add_edge(START, "research")
    builder.add_edge("research", "write_draft")
    builder.add_edge("write_draft", "await_review")
    builder.add_conditional_edges(
        "await_review", route_after_review, {"revise": "revise", END: "halt_check"}
    )
    builder.add_conditional_edges(
        "revise", route_after_revision, {"await_review": "await_review", END: "halt_check"}
    )
    builder.add_edge("halt_check", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
