"""Graph transition tests: the HITL pause, approval, revision loop and its ceiling."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from digital_economy_agent.agents import build_graph

CONFIG = {"configurable": {"thread_id": "test-thread"}}


def _initial(question: str = "Summarise the shift to Tech for Good", **kw):
    return {"question": question, "top_k": 3, "max_revisions": 2, **kw}


async def test_pauses_at_human_review(gateway, retriever):
    """The graph must stop at the review gate and expose the draft for inspection."""
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    out = await app.ainvoke(_initial(), config=CONFIG)

    interrupts = out["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["draft"]
    assert payload["revisions_used"] == 0
    assert payload["revisions_allowed"] == 2
    # Paused, not finished: no final report exists yet.
    assert not out.get("final_report")
    assert app.get_state(CONFIG).next == ("await_review",)


async def test_approval_finalises_the_draft(gateway, retriever):
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    paused = await app.ainvoke(_initial(), config=CONFIG)
    draft = paused["__interrupt__"][0].value["draft"]

    final = await app.ainvoke(Command(resume={"action": "approve"}), config=CONFIG)

    assert final["final_report"] == draft
    assert final["decision"] == "approve"
    assert final.get("revisions", 0) == 0
    assert app.get_state(CONFIG).next == ()


async def test_revision_loops_back_and_returns_to_review(gateway, retriever):
    """A revise decision must produce a new draft and pause for review again."""
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    paused = await app.ainvoke(_initial(), config=CONFIG)
    first_draft = paused["__interrupt__"][0].value["draft"]

    again = await app.ainvoke(
        Command(resume={"action": "revise", "feedback": "Add Vietnam talent detail."}),
        config=CONFIG,
    )

    payload = again["__interrupt__"][0].value
    assert payload["revisions_used"] == 1
    assert payload["draft"] != first_draft
    assert again["feedback_history"] == ["Add Vietnam talent detail."]


async def test_revision_budget_halts_the_loop(gateway, retriever):
    """
    The loop must be bounded. v1's Flowise loop node had no ceiling, so a reviewer who
    kept requesting changes could spend tokens indefinitely.
    """
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    await app.ainvoke(_initial(max_revisions=2), config=CONFIG)

    revise = Command(resume={"action": "revise", "feedback": "More detail."})
    await app.ainvoke(revise, config=CONFIG)  # revision 1 -> pauses again
    final = await app.ainvoke(revise, config=CONFIG)  # revision 2 -> budget exhausted

    assert final["revisions"] == 2
    assert "__interrupt__" not in final
    assert "revision budget exhausted" in final["halted_reason"]
    # The latest draft is still returned rather than discarded.
    assert final["final_report"]
    assert app.get_state(CONFIG).next == ()


async def test_empty_retrieval_is_reported_not_hidden(gateway, retriever):
    """
    With no matching context the agent must say so and make no LLM call for findings.

    Silently writing a report from zero context is the failure mode v1's groundedness
    metric was structurally unable to detect.
    """
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    out = await app.ainvoke(_initial(question="zzzz qqqq xxxx"), config=CONFIG)

    assert out["retrieved_chars"] == 0
    assert out["citations"] == []
    assert "No relevant context" in out["findings"]
    # Only the drafting call was made; research made none.
    assert [c["node"] for c in out["llm_calls"]] == ["write_draft"]


async def test_revision_without_feedback_is_rejected(gateway, retriever):
    """An unguided rewrite wastes an iteration, so feedback is contractually required."""
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    await app.ainvoke(_initial(), config=CONFIG)

    with pytest.raises(Exception, match="feedback is required"):
        await app.ainvoke(Command(resume={"action": "revise", "feedback": "   "}), config=CONFIG)


async def test_cost_and_prompt_versions_are_recorded(gateway, retriever):
    """Every model call must be attributable to a model and a pinned prompt version."""
    app = build_graph(gateway, retriever, checkpointer=InMemorySaver())
    paused = await app.ainvoke(_initial(), config=CONFIG)
    await app.ainvoke(Command(resume={"action": "approve"}), config=CONFIG)

    calls = paused["llm_calls"]
    assert [c["node"] for c in calls] == ["research", "write_draft"]
    assert all(c["prompt_version"].endswith(".v1") for c in calls)
    assert all(c["model"] == "fake/primary" for c in calls)
    # 100 in + 50 out at 0.10/0.40 per million.
    assert calls[0]["cost_usd"] == pytest.approx((100 * 0.10 + 50 * 0.40) / 1_000_000)
    assert gateway.stats.requests == 2
    assert gateway.stats.total_cost_usd > 0
