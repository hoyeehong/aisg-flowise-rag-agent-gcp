"""
The event path end to end, against a real Postgres with pgvector.

The interesting properties are all about repetition and ordering -- what happens when
the same message arrives twice, or an older version arrives after a newer one -- and
those live in SQL (`ON CONFLICT`, the conditional `DO UPDATE`, the tenant policy).
A fake store would test none of them.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from digital_economy_agent.messaging import (
    DocumentIngestionHandler,
    InMemoryDeadLetters,
    InMemorySubscriber,
    MessageConsumer,
)
from digital_economy_agent.retrieval import EmbedTask, HashingEmbedder, PgVectorStore

DSN = os.getenv("TEST_POSTGRES_DSN", "postgresql://postgres:devpass@localhost:55432/agent")
APP_ROLE = "agent_app"
APP_PASSWORD = "apppass"


def _app_dsn(admin_dsn: str) -> str:
    info = conninfo_to_dict(admin_dsn)
    info["user"] = APP_ROLE
    info["password"] = APP_PASSWORD
    return make_conninfo(**info)


APP_DSN = _app_dsn(DSN)


def _postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason=f"no Postgres at {DSN}")


class CountingEmbedder:
    """
    Wraps the deterministic embedder and counts how many texts it was asked to embed.

    The count is the point. Embedding is the expensive, rate-limited, billable step, so
    "did this redelivery re-embed?" is a cost question that a passing ingest cannot
    answer on its own.
    """

    def __init__(self) -> None:
        self._inner = HashingEmbedder()
        self.embedded = 0
        self.calls = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: list[str], *, task: EmbedTask) -> list[list[float]]:
        self.calls += 1
        self.embedded += len(texts)
        return await self._inner.embed(texts, task=task)


@pytest.fixture
def tenant() -> Iterator[str]:
    name = f"evt-{uuid.uuid4().hex[:12]}"
    yield name
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE tenant_id LIKE %s", (f"{name}%",))
            conn.commit()
    except Exception:
        pass


@pytest.fixture
async def store() -> PgVectorStore:
    admin = PgVectorStore(DSN)
    await admin.ensure_schema()
    await admin.ensure_app_role(APP_ROLE, password=APP_PASSWORD)
    return PgVectorStore(APP_DSN)


def _event(tenant: str, *, text: str = "Digital trust and governance in the region.", **extra):
    payload = {
        "tenant_id": tenant,
        "source": "events.pdf",
        "pages": [{"page": 1, "text": text}],
    }
    payload.update(extra)
    return payload


async def _consume(sub, store, embedder, *, dlq=None, attempts=3):
    handler = DocumentIngestionHandler(store, embedder)
    consumer = MessageConsumer(sub, handler, dead_letters=dlq, max_delivery_attempts=attempts)
    return await consumer.run_once()


# --- the happy path --------------------------------------------------------


async def test_a_document_event_is_ingested(store, tenant):
    sub = InMemorySubscriber()
    sub.publish(_event(tenant))
    embedder = CountingEmbedder()

    outcome = await _consume(sub, store, embedder)

    assert outcome.processed == 1
    assert await store.count(tenant_id=tenant) > 0
    assert embedder.embedded > 0


async def test_an_event_only_writes_its_own_tenant(store, tenant):
    """The policy covers the event path too: nothing lands anywhere else."""
    sub = InMemorySubscriber()
    sub.publish(_event(tenant))
    await _consume(sub, store, CountingEmbedder())

    assert await store.count(tenant_id=tenant) > 0
    assert await store.count(tenant_id=f"{tenant}-other") == 0


# --- redelivery ------------------------------------------------------------


async def test_a_redelivered_message_does_not_pay_for_embeddings_again(store, tenant):
    """
    At-least-once delivery means this *will* happen, so the cost of it matters.

    Content hashing already made the storage idempotent, but the hash used to be
    computed at insert time -- so a redelivery embedded every chunk and then upserted
    to no effect. On a metered, rate-limited provider that is a recurring bill.
    """
    embedder = CountingEmbedder()

    first_sub = InMemorySubscriber()
    first_sub.publish(_event(tenant), message_id="dup")
    await _consume(first_sub, store, embedder)
    after_first = embedder.embedded
    assert after_first > 0

    # The identical message again, as a broker redelivery would deliver it.
    second_sub = InMemorySubscriber()
    second_sub.publish(_event(tenant), message_id="dup")
    outcome = await _consume(second_sub, store, embedder)

    assert outcome.processed == 1, "a redelivery must still be acknowledged"
    assert embedder.embedded == after_first, (
        f"re-embedded {embedder.embedded - after_first} chunks on a redelivery"
    )
    assert await store.count(tenant_id=tenant) == await store.count(tenant_id=tenant)


async def test_changed_text_is_embedded_but_unchanged_text_is_not(store, tenant):
    """Dedup must not be so eager that a genuine edit is skipped."""
    embedder = CountingEmbedder()
    sub = InMemorySubscriber()
    sub.publish(_event(tenant, text="Original text about digital trust."))
    await _consume(sub, store, embedder)
    baseline = embedder.embedded

    changed = InMemorySubscriber()
    changed.publish(_event(tenant, text="Completely different text about talent gaps."))
    await _consume(changed, store, embedder)

    assert embedder.embedded > baseline, "an edited document must be re-embedded"


# --- ordering --------------------------------------------------------------


async def test_an_older_version_does_not_overwrite_the_same_chunk(store, tenant):
    """
    A late-arriving older version must not replace a newer row for the same chunk.

    Out-of-order delivery is normal: Kafka orders per partition, Pub/Sub only with an
    ordering key. Without the guard the older delivery's embedding, redaction report
    and version stamp would overwrite the newer ones.

    Note the scope. This protects a row whose *content is identical*, which is the only
    case that reaches ON CONFLICT -- edited text hashes differently and inserts as its
    own row. Chunks a newer version deleted are therefore **not** removed by this; see
    test_a_superseded_version_leaves_its_chunks_behind below.
    """
    embedder = CountingEmbedder()
    newer = datetime.now(UTC)
    older = newer - timedelta(hours=1)
    body = "A paragraph that does not change between versions."

    sub = InMemorySubscriber()
    sub.publish(_event(tenant, text=body, source_updated_at=newer.isoformat()))
    await _consume(sub, store, embedder)

    stale = InMemorySubscriber()
    stale.publish(_event(tenant, text=body, source_updated_at=older.isoformat()))
    outcome = await _consume(stale, store, embedder)

    # Handled and acknowledged -- being stale is not an error.
    assert outcome.processed == 1

    async with store._session(tenant) as cur:
        await cur.execute("SELECT source_updated_at FROM chunks WHERE tenant_id = %s", (tenant,))
        stamps = [r["source_updated_at"] for r in await cur.fetchall()]
    assert stamps, "the document vanished"
    assert all(s == newer for s in stamps), (
        f"the version stamp went backwards: {stamps} should all be {newer}"
    )


async def test_a_superseded_version_leaves_its_chunks_behind(store, tenant):
    """
    A known limitation, pinned so it cannot regress silently into a surprise.

    An older version redelivered after a newer one re-inserts the chunks it contains.
    Those are a different content hash, so they do not conflict with anything -- they
    are added, and the corpus then holds text that is not in the current document.

    Fixing this needs document reconciliation: after ingesting a version, delete the
    rows for that (tenant, source) that the version does not contain. That is not
    implemented, and the reason is `ingest_pdf(max_pages=N)`: partial ingestion is a
    supported path, so pruning "everything not in this batch" would delete the rest of
    the document. It needs an explicit "this is the complete document" signal, which is
    a design decision rather than a patch -- and deletion is the wrong direction to
    guess in.
    """
    embedder = CountingEmbedder()
    newer = datetime.now(UTC)
    older = newer - timedelta(hours=1)

    current = InMemorySubscriber()
    current.publish(
        _event(
            tenant, text="The current wording of this section.", source_updated_at=newer.isoformat()
        )
    )
    await _consume(current, store, embedder)

    superseded = InMemorySubscriber()
    superseded.publish(
        _event(
            tenant,
            text="Wording that a later version removed.",
            source_updated_at=older.isoformat(),
        )
    )
    await _consume(superseded, store, embedder)

    removed = await store.lexical_search("removed", top_k=5, tenant_id=tenant)
    assert removed, (
        "document reconciliation appears to have been implemented -- if so, update this "
        "test and the README limitation rather than deleting the assertion"
    )
    assert await store.lexical_search("current", top_k=5, tenant_id=tenant), (
        "the current version must still be present"
    )


# --- failure handling ------------------------------------------------------


async def test_a_malformed_event_is_dead_lettered_without_retrying(store, tenant):
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters()
    sub.publish(b"{ not a document event", message_id="broken")

    outcome = await _consume(sub, store, CountingEmbedder(), dlq=dlq)

    assert outcome.dead_lettered == 1
    assert outcome.retried == 0
    assert dlq.ids == ["broken"]
    assert await store.count(tenant_id=tenant) == 0


async def test_an_event_naming_an_invalid_tenant_never_reaches_the_database(store):
    """A tenant that fails validation must be retired, not written under some fallback."""
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters()
    sub.publish(_event("../etc/passwd"), message_id="bad-tenant")

    outcome = await _consume(sub, store, CountingEmbedder(), dlq=dlq)

    assert outcome.dead_lettered == 1
    assert "not a document event" in dlq.messages[0][1]
