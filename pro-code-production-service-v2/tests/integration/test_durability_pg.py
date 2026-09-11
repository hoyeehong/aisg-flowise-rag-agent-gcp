"""
Durability tests against a real AsyncPostgresSaver.

These exist because Phase 2 shipped a broken Postgres checkpointer path. The whole
suite passed, because every other test uses InMemorySaver -- which tolerates the
synchronous ``get_state`` call that AsyncPostgresSaver rejects with InvalidStateError.
The bug only appeared when the built container was driven end to end.

The lesson encoded here: a checkpointer is not interchangeable with its in-memory
stand-in, so the durable one needs its own coverage.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from digital_economy_agent.agents import build_graph
from digital_economy_agent.api.service import ReportService, RunNotFoundError
from digital_economy_agent.gateway import ModelGateway, ModelSpec

from ..conftest import FakeProvider

DSN = os.getenv("TEST_POSTGRES_DSN", "postgresql://postgres:devpass@localhost:55432/agent")


# Runs are owned by a tenant, and reads verify ownership.
TENANT = "durability-test"


def _postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason=f"no Postgres at {DSN}")


def _gateway(no_sleep) -> ModelGateway:
    return ModelGateway(
        [ModelSpec(provider="fake", model="primary")],
        {"fake": FakeProvider("fake")},
        sleep=no_sleep,
    )


async def _service(retriever, no_sleep) -> tuple[ReportService, AsyncPostgresSaver]:
    """A service backed by a genuine AsyncPostgresSaver, as production runs."""
    saver_cm = AsyncPostgresSaver.from_conn_string(DSN)
    saver = await saver_cm.__aenter__()
    await saver.setup()
    graph = build_graph(_gateway(no_sleep), retriever, checkpointer=saver)
    return ReportService(graph, default_top_k=3, max_revisions=2), saver_cm


async def test_state_reads_work_against_the_async_saver(retriever, no_sleep):
    """
    The regression test for the shipped bug.

    ReportService.get() used the synchronous graph.get_state(), which raises
    InvalidStateError on AsyncPostgresSaver from the main thread. Every read -- start,
    review and get -- returned a 500 against a real database while the suite was green.
    """
    service, cm = await _service(retriever, no_sleep)
    try:
        thread_id = f"dur-{uuid.uuid4().hex[:12]}"
        state = await service.start(
            thread_id, "Summarise digital trust in the region", tenant_id=TENANT
        )
        assert state.status == "awaiting_review"
        assert state.draft

        # The call that used to raise.
        again = await service.get(thread_id, tenant_id=TENANT)
        assert again.thread_id == thread_id
        assert again.status == "awaiting_review"
        assert again.draft == state.draft
    finally:
        await cm.__aexit__(None, None, None)


async def test_paused_run_survives_a_new_service_instance(retriever, no_sleep):
    """
    The point of the Postgres checkpointer.

    A second ReportService with a fresh saver stands in for a restarted process: it
    shares no memory with the first, so if the pause is readable the state genuinely
    lives in Postgres rather than in the process.
    """
    first, cm1 = await _service(retriever, no_sleep)
    try:
        thread_id = f"dur-{uuid.uuid4().hex[:12]}"
        created = await first.start(
            thread_id, "Summarise digital trust in the region", tenant_id=TENANT
        )
    finally:
        await cm1.__aexit__(None, None, None)

    second, cm2 = await _service(retriever, no_sleep)
    try:
        recovered = await second.get(thread_id, tenant_id=TENANT)
        assert recovered.status == "awaiting_review"
        assert recovered.draft == created.draft

        # And a reviewer can still act on it after the "restart".
        final = await second.review(thread_id, "approve", "", tenant_id=TENANT)
        assert final.status == "completed"
        assert final.final_report == created.draft
    finally:
        await cm2.__aexit__(None, None, None)


async def test_revision_history_survives_too(retriever, no_sleep):
    """Feedback accumulated before a restart must not be lost."""
    first, cm1 = await _service(retriever, no_sleep)
    try:
        thread_id = f"dur-{uuid.uuid4().hex[:12]}"
        await first.start(thread_id, "Summarise digital talent gaps", tenant_id=TENANT)
        await first.review(thread_id, "revise", "Add Vietnam specifics.", tenant_id=TENANT)
    finally:
        await cm1.__aexit__(None, None, None)

    second, cm2 = await _service(retriever, no_sleep)
    try:
        state = await second.get(thread_id, tenant_id=TENANT)
        assert state.revisions_used == 1
        assert state.feedback_history == ["Add Vietnam specifics."]
    finally:
        await cm2.__aexit__(None, None, None)


async def test_unknown_thread_still_raises_not_found(retriever, no_sleep):
    """A missing checkpoint must be a clean 404, not a driver-level error."""
    service, cm = await _service(retriever, no_sleep)
    try:
        with pytest.raises(RunNotFoundError):
            await service.get(f"absent-{uuid.uuid4().hex[:12]}", tenant_id=TENANT)
    finally:
        await cm.__aexit__(None, None, None)
