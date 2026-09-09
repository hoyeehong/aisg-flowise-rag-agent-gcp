"""
Integration tests against a real Postgres with pgvector.

Deliberately not mocked. The interesting behaviour lives in SQL — the generated
tsvector column, HNSW cosine ordering, and ``ON CONFLICT ... RETURNING (xmax = 0)`` for
idempotency — none of which a fake would exercise. A stub here would test nothing.

Embeddings use the deterministic hashing embedder so these tests need no API key and
no quota; the semantic quality of a real embedder is measured separately.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from digital_economy_agent.ingestion import PatternRedactor, ingest_pages
from digital_economy_agent.retrieval import (
    ChunkingConfig,
    EmbedTask,
    HashingEmbedder,
    HybridConfig,
    HybridRetriever,
    PgVectorStore,
)
from digital_economy_agent.tools import RetrievalRequest
from digital_economy_agent.tools.types import Chunk

DSN = os.getenv("TEST_POSTGRES_DSN", "postgresql://postgres:devpass@localhost:55432/agent")

# Tests query as the least-privilege application role, not as the administrative role
# in TEST_POSTGRES_DSN. This is the difference between testing row-level security and
# testing around it: a superuser ignores policies entirely, so the first version of
# this schema was enabled, forced, and inert -- an unscoped count returned 299 rows
# with all 19 of these tests green. Credentials are local-container only.
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

CORPUS = [
    Chunk(
        text="Digital trust and governance require proactive cybersecurity, consumer "
        "safety, privacy protection and responsible AI frameworks.",
        source="t.pdf",
        page=18,
    ),
    Chunk(
        text="ASEAN DEFA harmonises cross-border data flows, digital payments and "
        "regulatory cohesion across member states.",
        source="t.pdf",
        page=31,
    ),
    Chunk(
        text="Digital talent supply constrains Indonesia, Vietnam and the Philippines, "
        "where demand for advanced skills outpaces the training pipeline.",
        source="t.pdf",
        page=23,
    ),
    Chunk(
        text="Southeast Asia is shifting from Tech for Growth to Tech for Good, "
        "prioritising inclusion and sustainability alongside GMV expansion.",
        source="t.pdf",
        page=4,
    ),
]


@pytest.fixture
def tenant() -> Iterator[str]:
    """
    A unique tenant per test, deleted afterwards.

    Cleanup matters: without it every run left its rows behind, and a local database
    accumulated 130+ dead tenants over the course of Phase 2 and 3. That is slow, and
    it makes ad-hoc inspection of the table useless.

    Runs on the administrative connection on purpose. An unscoped ``DELETE`` as the
    application role now matches zero rows -- the policy failing closed, which is the
    behaviour being tested elsewhere in this file rather than a problem to work around.
    """
    name = f"test-{uuid.uuid4().hex[:12]}"
    yield name
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE tenant_id = %s", (name,))
            conn.commit()
    except Exception:
        # Cleanup failure must not fail a passing test; the next run reuses no ids.
        pass


@pytest.fixture
async def admin_store() -> PgVectorStore:
    """
    Store on the administrative connection. DDL and role management need ownership,
    which the application role deliberately does not have.
    """
    s = PgVectorStore(DSN)
    await s.ensure_schema()
    await s.ensure_app_role(APP_ROLE, password=APP_PASSWORD)
    return s


@pytest.fixture
async def store(admin_store: PgVectorStore) -> PgVectorStore:
    """The store as the service uses it: the app role, subject to the RLS policy."""
    return PgVectorStore(APP_DSN)


@pytest.fixture
async def seeded(store, tenant):
    emb = HashingEmbedder()
    vectors = await emb.embed([c.text for c in CORPUS], task=EmbedTask.DOCUMENT)
    await store.upsert(CORPUS, vectors, embedder=emb.name, tenant_id=tenant)
    return store, emb, tenant


# --- tenant isolation (row-level security) ---------------------------------


async def test_isolation_is_enforced_for_the_app_role(store):
    """
    The app role must be genuinely subject to the policy, not merely covered by one.

    Asserting ``enforced`` rather than ``rls_enabled`` is the whole point: the first
    version of this schema had RLS enabled and forced, a correct policy, and no
    enforcement at all, because the connection was a superuser.
    """
    status = await store.isolation_status()
    assert status["enforced"] is True, status
    assert status["is_superuser"] is False
    assert status["bypasses_rls"] is False
    assert status["unscoped_visible_rows"] == 0


async def test_superuser_connection_reports_isolation_unenforced(admin_store, seeded):
    """
    Regression test for the bug this feature shipped with.

    A superuser bypasses RLS, so isolation is not in force on that connection even
    though every table-level fact looks right. The probe must say so instead of
    reporting the configuration and letting the reader infer enforcement.
    """
    status = await admin_store.isolation_status()
    assert status["is_superuser"] is True
    assert status["rls_enabled"] is True
    assert status["policy_present"] is True
    # Configuration correct, enforcement absent -- exactly the state that passed tests.
    assert status["enforced"] is False, status


async def test_unscoped_connection_reads_nothing(seeded):
    """A connection that never sets app.tenant_id sees zero rows, not every row."""
    import psycopg

    _, _, tenant = seeded
    async with await psycopg.AsyncConnection.connect(APP_DSN) as conn, conn.cursor() as cur:
        await cur.execute("SELECT count(*) FROM chunks")
        assert (await cur.fetchone())[0] == 0
        # Naming the tenant in the WHERE clause does not help: the policy still has no
        # tenant to compare against, so the predicate is NULL for every row.
        await cur.execute("SELECT count(*) FROM chunks WHERE tenant_id = %s", (tenant,))
        assert (await cur.fetchone())[0] == 0


async def test_policy_and_not_the_where_clause_is_the_boundary(seeded, store):
    """
    With tenant A set, asking explicitly for tenant B returns nothing.

    This distinguishes the policy from the application's own ``WHERE tenant_id = %s``.
    If the filter were doing the work, this query would return B's rows.
    """
    import psycopg

    _, emb, tenant = seeded
    other = f"{tenant}-other"
    vectors = await emb.embed([CORPUS[0].text], task=EmbedTask.DOCUMENT)
    await store.upsert(CORPUS[:1], vectors, embedder=emb.name, tenant_id=other)
    try:
        async with await psycopg.AsyncConnection.connect(APP_DSN) as conn, conn.cursor() as cur:
            await cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
            await cur.execute("SELECT count(*) FROM chunks WHERE tenant_id = %s", (other,))
            assert (await cur.fetchone())[0] == 0
    finally:
        import psycopg as _pg

        with _pg.connect(DSN) as c, c.cursor() as cur2:
            cur2.execute("DELETE FROM chunks WHERE tenant_id = %s", (other,))
            c.commit()


async def test_cross_tenant_write_is_refused(seeded):
    """
    WITH CHECK stops a scoped connection writing into another tenant.

    Without it, isolation would be read-only: a caller could not see tenant B but
    could still insert rows attributed to it.
    """
    import psycopg

    _, _, tenant = seeded
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        async with await psycopg.AsyncConnection.connect(APP_DSN) as conn, conn.cursor() as cur:
            await cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
            await cur.execute(
                """
                INSERT INTO chunks (content_hash, tenant_id, source, page, text,
                                    embedding, embedder)
                VALUES ('x', %s, 's.pdf', 1, 'x', %s, 'e')
                """,
                (f"{tenant}-other", "[" + ",".join(["0.1"] * 768) + "]"),
            )


async def test_store_rejects_an_empty_tenant(store):
    """
    An empty tenant must fail loudly rather than being set as an empty GUC.

    ``set_config('app.tenant_id', '', true)`` would compare equal to no real tenant,
    so the query would silently return nothing -- a wrong answer dressed as an empty
    result, which is harder to notice than an exception.
    """
    with pytest.raises(ValueError, match="non-empty"):
        await store.count(tenant_id="")


# --- storage ---------------------------------------------------------------


async def test_ensure_schema_is_idempotent(admin_store):
    await admin_store.ensure_schema()
    await admin_store.ensure_schema()  # must not raise on a second application


async def test_bootstrap_works_before_the_extension_exists():
    """
    Schema creation must not require the pgvector type adapter.

    Registering the adapter queries the database for the `vector` OID, which only
    exists after CREATE EXTENSION -- so a store that registers on every connection
    cannot bootstrap a fresh database at all. This test only proves the non-registering
    connection path works; the genuine regression test is CI, whose Postgres service
    container starts empty on every run. Do not "optimise" that by caching the
    database volume, or this bug becomes invisible again.
    """
    fresh = PgVectorStore(DSN)
    conn = await fresh._connect(register_vector=False)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 AS ok")
            assert (await cur.fetchone())["ok"] == 1
    finally:
        await conn.close()
    await fresh.ensure_schema()


async def test_upsert_reports_inserts_then_updates(store, tenant):
    emb = HashingEmbedder()
    vectors = await emb.embed([c.text for c in CORPUS], task=EmbedTask.DOCUMENT)

    inserted, updated = await store.upsert(CORPUS, vectors, embedder=emb.name, tenant_id=tenant)
    assert (inserted, updated) == (len(CORPUS), 0)

    inserted, updated = await store.upsert(CORPUS, vectors, embedder=emb.name, tenant_id=tenant)
    assert (inserted, updated) == (0, len(CORPUS)), "re-ingest must update, not duplicate"
    assert await store.count(tenant_id=tenant) == len(CORPUS)


async def test_tenants_are_isolated(store, tenant):
    emb = HashingEmbedder()
    vectors = await emb.embed([c.text for c in CORPUS[:2]], task=EmbedTask.DOCUMENT)
    await store.upsert(CORPUS[:2], vectors, embedder=emb.name, tenant_id=tenant)

    assert await store.count(tenant_id=tenant) == 2
    assert await store.count(tenant_id=f"{tenant}-other") == 0

    query = (await emb.embed(["cybersecurity"], task=EmbedTask.QUERY))[0]
    assert await store.vector_search(query, top_k=5, tenant_id=f"{tenant}-other") == []


async def test_embedding_width_mismatch_is_rejected(store, tenant):
    """A wrong-width vector must fail loudly here, not as an opaque SQL error."""
    with pytest.raises(ValueError, match="wrong width"):
        await store.upsert(CORPUS[:1], [[0.1, 0.2]], embedder="bad", tenant_id=tenant)


async def test_count_mismatch_is_rejected(store, tenant):
    with pytest.raises(ValueError, match="mismatch"):
        await store.upsert(CORPUS, [[0.0] * 768], embedder="bad", tenant_id=tenant)


# --- search ----------------------------------------------------------------


async def test_lexical_search_finds_exact_identifiers(seeded):
    """
    The half vectors are bad at.

    An embedding blurs 'ASEAN DEFA' toward generic regional-cooperation language; the
    tsvector index matches the token that makes the result correct.
    """
    store, _, tenant = seeded
    hits = await store.lexical_search("ASEAN DEFA", top_k=5, tenant_id=tenant)
    assert hits, "expected a lexical match"
    assert hits[0].chunk.page == 31


async def test_lexical_search_returns_empty_for_absent_terms(seeded):
    store, _, tenant = seeded
    assert await store.lexical_search("maritime shipping tariffs", top_k=5, tenant_id=tenant) == []


async def test_lexical_search_reduces_natural_language_queries(seeded):
    """
    A question-form query must still match lexically.

    ``websearch_to_tsquery`` ANDs bare terms, so passing a whole question required a
    single chunk to contain every word including "how", "does" and "report" -- which
    essentially never happens. The lexical half of hybrid retrieval therefore returned
    zero hits for every real query, making "hybrid" search vector search with extra
    steps. The Phase 3 eval harness caught it on its first live run.
    """
    store, _, tenant = seeded
    hits = await store.lexical_search(
        "How does the report address digital trust and cybersecurity?",
        top_k=5,
        tenant_id=tenant,
    )
    assert hits, "a question-form query must not return zero lexical hits"
    assert 18 in [h.chunk.page for h in hits]


async def test_lexical_terms_are_disjunctive(seeded):
    """Any content term may match; requiring all of them is what broke question queries."""
    store, _, tenant = seeded
    # 'cybersecurity' and 'talent' never co-occur in a chunk of this corpus.
    hits = await store.lexical_search("cybersecurity talent", top_k=5, tenant_id=tenant)
    pages = {h.chunk.page for h in hits}
    assert pages & {18, 23}, f"expected a match on either term, got {pages}"


async def test_lexical_search_ignores_framing_words(seeded):
    """Interrogatives carry no signal and, when ANDed, actively suppress matches."""
    from digital_economy_agent.retrieval.store import lexical_query_terms

    reduced = lexical_query_terms("What does the report say about digital talent?")
    assert "digital" in reduced and "talent" in reduced
    for noise in ("what", "does", "the", "report", "say", "about"):
        assert noise not in reduced.split(" OR "), f"{noise!r} should be dropped"


async def test_vector_search_orders_by_similarity(seeded):
    store, emb, tenant = seeded
    query = (await emb.embed(["cybersecurity privacy responsible AI"], task=EmbedTask.QUERY))[0]
    hits = await store.vector_search(query, top_k=4, tenant_id=tenant)
    assert hits[0].chunk.page == 18
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].rank == 1


# --- hybrid ----------------------------------------------------------------


async def test_hybrid_recall_beats_the_v1_baseline(seeded):
    """
    The measured Phase 2 objective.

    v1's token-overlap retriever matched 1 of 4 chunks for this query and returned 177
    characters of context -- too little to ground a report, which is exactly why Phase
    0 had to report groundedness as INCONCLUSIVE.
    """
    store, emb, tenant = seeded
    retriever = HybridRetriever(store, emb, config=HybridConfig(), tenant_id=tenant)
    result = await retriever.retrieve(
        RetrievalRequest(
            query="Summarise the shift from Tech for Growth to Tech for Good in Southeast Asia",
            top_k=4,
        )
    )
    assert len(result.chunks) == 4, "hybrid must recall the whole corpus, not 1 of 4"
    assert result.total_chars > 400
    assert result.chunks[0].page == 4, "the directly relevant passage should rank first"


async def test_hybrid_respects_top_k(seeded):
    store, emb, tenant = seeded
    retriever = HybridRetriever(store, emb, config=HybridConfig(), tenant_id=tenant)
    result = await retriever.retrieve(RetrievalRequest(query="digital", top_k=2))
    assert len(result.chunks) <= 2


async def test_hybrid_returns_empty_for_an_empty_tenant(store, tenant):
    """An empty result must be returned cleanly; the graph reports it rather than hiding it."""
    retriever = HybridRetriever(store, HashingEmbedder(), tenant_id=f"{tenant}-empty")
    result = await retriever.retrieve(RetrievalRequest(query="anything", top_k=5))
    assert result.chunks == []
    assert result.total_chars == 0


async def test_hybrid_carries_citations(seeded):
    store, emb, tenant = seeded
    retriever = HybridRetriever(store, emb, tenant_id=tenant)
    result = await retriever.retrieve(RetrievalRequest(query="digital talent", top_k=2))
    assert all(c.citation().startswith("t.pdf, p.") for c in result.chunks)
    assert "[t.pdf, p." in result.context


# --- ingestion end to end --------------------------------------------------


async def test_ingest_pages_redacts_before_storage(store, tenant):
    """
    PII must never reach the embedding provider or the vector column.

    Redacting after embedding would leave the original recoverable in the vector's
    neighbourhood, and would already have sent it to a third party.
    """
    pages = [
        (1, "Contact the analyst at officer@agency.gov.sg or on 91234567 about the policy."),
        (2, "The NRIC S1234567D appears in the appendix alongside digital trust guidance."),
    ]
    result = await ingest_pages(
        pages,
        source="pii.pdf",
        store=store,
        embedder=HashingEmbedder(),
        redactor=PatternRedactor(),
        chunking=ChunkingConfig(chunk_size=400, overlap=50, min_chunk_chars=20),
        tenant_id=tenant,
    )
    assert result.redactions.counts["email"] == 1
    assert result.redactions.counts["nric_fin"] == 1
    assert result.inserted == result.chunks > 0

    # Read every stored row back via vector search, which always returns rows when the
    # tenant is non-empty. A lexical query could return nothing and make the assertions
    # below pass vacuously -- unacceptable in a test whose whole point is that PII is
    # absent, so the non-empty check comes first.
    probe = (await HashingEmbedder().embed(["policy"], task=EmbedTask.QUERY))[0]
    stored = await store.vector_search(probe, top_k=50, tenant_id=tenant)
    body = " ".join(h.chunk.text for h in stored)
    assert body, "read-back returned nothing; the PII assertions would be vacuous"
    assert len(stored) == result.chunks

    assert "officer@agency.gov.sg" not in body
    assert "91234567" not in body
    assert "S1234567D" not in body
    assert "[EMAIL_REDACTED]" in body
    assert "[NRIC_REDACTED]" in body


async def test_ingest_rerun_is_idempotent(store, tenant):
    pages = [(1, "Digital trust guidance for the region, repeated ingestion test corpus.")]
    kwargs = dict(
        source="idem.pdf",
        store=store,
        embedder=HashingEmbedder(),
        chunking=ChunkingConfig(chunk_size=400, overlap=50, min_chunk_chars=20),
        tenant_id=tenant,
    )
    first = await ingest_pages(pages, **kwargs)
    second = await ingest_pages(pages, **kwargs)

    assert first.inserted > 0
    assert second.inserted == 0
    assert second.idempotent_rerun is True
    assert await store.count(tenant_id=tenant) == first.chunks


async def test_ingest_records_provenance(store, tenant):
    result = await ingest_pages(
        [(1, "Provenance test text about digital governance and regional policy.")],
        source="prov.pdf",
        store=store,
        embedder=HashingEmbedder(),
        chunking=ChunkingConfig(chunk_size=500, overlap=100, min_chunk_chars=20),
        tenant_id=tenant,
    )
    assert result.embedder == "hashing@768"
    assert result.redactor == "pattern-redactor"
    assert result.chunking == {"chunk_size": 500, "overlap": 100}
