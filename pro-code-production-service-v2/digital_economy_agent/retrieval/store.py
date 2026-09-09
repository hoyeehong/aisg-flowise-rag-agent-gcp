"""
pgvector-backed chunk store: schema management, idempotent upsert, and the two
halves of hybrid search.

Postgres rather than a dedicated vector database because the corpus also needs
relational filters (tenant, source, page) and full-text search. Running one engine for
vectors, lexical search and metadata avoids keeping two stores consistent, which is a
larger operational cost than the marginal vector performance difference at this scale.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..tools.types import Chunk

_SCHEMA = Path(__file__).parent / "schema.sql"

_TERM = re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b")

# Interrogatives and framing words carry no retrieval signal but, under conjunctive
# tsquery semantics, every one of them must appear in a chunk for it to match.
_QUERY_STOPWORDS = frozenset(
    [
        "about",
        "across",
        "after",
        "against",
        "all",
        "also",
        "and",
        "any",
        "are",
        "around",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "describe",
        "describes",
        "did",
        "does",
        "doing",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "how",
        "identified",
        "into",
        "its",
        "itself",
        "more",
        "most",
        "only",
        "other",
        "over",
        "own",
        "report",
        "reports",
        "said",
        "same",
        "say",
        "says",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "would",
        "you",
        "your",
    ]
)


def lexical_query_terms(query: str) -> str:
    """
    Reduce a natural-language question to an OR-joined keyword query.

    ``websearch_to_tsquery`` ANDs bare terms, so passing a whole question requires a
    single chunk to contain every word including "how", "does" and "report" -- which
    essentially never happens. Before this, the lexical half of hybrid retrieval
    returned zero hits for every question-form query, so "hybrid" search was vector
    search with extra steps. Found by the Phase 3 eval harness on its first live run.
    """
    terms = [
        term
        for term in (m.group(0).lower() for m in _TERM.finditer(query))
        if term not in _QUERY_STOPWORDS
    ]
    # Deduplicate while preserving order, then OR so any content term can match.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return " OR ".join(unique)


def content_hash(source: str, page: int | None, text: str) -> str:
    """
    Stable identity for a chunk.

    Includes source and page, not just text: identical boilerplate on two pages is two
    distinct citable chunks, and collapsing them would silently drop a citation.
    """
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(page if page is not None else -1).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the rank it achieved in one retrieval strategy."""

    chunk: Chunk
    score: float
    rank: int


class PgVectorStore:
    """Async pgvector store. Callers own the connection lifecycle via ``connect()``."""

    def __init__(self, dsn: str, *, dimensions: int = 768) -> None:
        self._dsn = dsn
        self.dimensions = dimensions

    async def _connect(self, *, register_vector: bool = True) -> psycopg.AsyncConnection[Any]:
        """
        Open a connection.

        ``register_vector`` must be False before the schema exists: registering the
        pgvector type adapter queries the database for the `vector` OID, which only
        exists after ``CREATE EXTENSION``. Bootstrapping a fresh database therefore has
        to connect without it -- otherwise ensure_schema() cannot create the very
        extension it needs, which is exactly what a fresh CI service container hits.
        """
        conn = await psycopg.AsyncConnection.connect(self._dsn, row_factory=dict_row)
        if register_vector:
            await register_vector_async(conn)
        return conn

    @asynccontextmanager
    async def _session(self, tenant_id: str) -> AsyncIterator[psycopg.AsyncCursor[Any]]:
        """
        A cursor inside a transaction whose row-level-security tenant is already set.

        Every tenant-scoped query goes through here, so the ``app.tenant_id`` setting
        the RLS policy reads can never be missing: the policy fails closed, meaning a
        forgotten SET returns zero rows rather than another tenant's.

        ``SET`` does not accept placeholders, so this uses ``set_config`` with
        ``is_local => true`` -- the function form of ``SET LOCAL``, scoped to this
        transaction. That scope matters more than it looks: the store currently opens a
        connection per call, but if a pool is introduced later, a session-scoped ``SET``
        would leak the previous request's tenant onto a recycled connection. That is the
        classic way RLS deployments are silently broken, and it is invisible to
        single-tenant testing.
        """
        if not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        conn = await self._connect()
        try:
            async with conn.transaction(), conn.cursor() as cur:
                await cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
                yield cur
        finally:
            await conn.close()

    async def ensure_schema(self) -> None:
        """Apply the DDL. Idempotent, so it is safe on every service start."""
        ddl = _SCHEMA.read_text(encoding="utf-8")
        if self.dimensions != 768:
            # Keep the declared column width and the embedder in lockstep; a mismatch
            # would otherwise surface as an opaque insert error much later.
            ddl = ddl.replace("VECTOR(768)", f"VECTOR({self.dimensions})")
        conn = await self._connect(register_vector=False)
        try:
            async with conn.cursor() as cur:
                await cur.execute(sql.SQL(ddl))
            await conn.commit()
        finally:
            await conn.close()

    async def ensure_app_role(self, role: str, *, password: str | None = None) -> None:
        """
        Create the least-privilege role the service connects as, and grant it exactly
        the DML it needs. Runs as an administrative role; idempotent.

        Enabling row-level security is not what enforces it. A superuser ignores
        policies outright, and ``FORCE ROW LEVEL SECURITY`` subjects only the table
        *owner* -- not a superuser. Connecting as ``postgres``, which local development
        and CI both did, leaves every policy inert while the whole suite stays green:
        measured directly on the first run of this schema, an unscoped
        ``SELECT count(*) FROM chunks`` returned 299 rows with RLS enabled and forced.

        ``NOSUPERUSER NOBYPASSRLS`` is therefore the load-bearing line here, not the
        GRANTs. It is re-applied on every call so that a role which later acquires
        BYPASSRLS out of band is brought back into the policy on the next start.
        """
        conn = await self._connect(register_vector=False)
        ident = sql.Identifier(role)
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 AS present FROM pg_roles WHERE rolname = %s", (role,))
                if await cur.fetchone() is None:
                    # Concurrent starts can both reach this branch; the loser gets a
                    # duplicate_object error, which is why callers run it once at
                    # bootstrap rather than per request.
                    await cur.execute(sql.SQL("CREATE ROLE {} LOGIN").format(ident))
                if password is not None:
                    await cur.execute(
                        sql.SQL("ALTER ROLE {} WITH PASSWORD {}").format(
                            ident, sql.Literal(password)
                        )
                    )
                await cur.execute(
                    sql.SQL("ALTER ROLE {} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE").format(
                        ident
                    )
                )
                await cur.execute(
                    sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO {}").format(ident)
                )
                # The BIGSERIAL primary key needs the sequence, or every INSERT fails
                # with a permission error that names the sequence, not the table.
                await cur.execute(
                    sql.SQL("GRANT USAGE, SELECT ON SEQUENCE chunks_id_seq TO {}").format(ident)
                )
            await conn.commit()
        finally:
            await conn.close()

    async def isolation_status(self) -> dict[str, Any]:
        """
        Report whether tenant isolation is actually being enforced on this connection.

        This exists because the first version of this schema was enabled, forced, and
        entirely ineffective: connected as a superuser, an unscoped
        ``SELECT count(*) FROM chunks`` returned 299 rows while every test passed.
        Enabling a policy and enforcing one are different facts, and only the second is
        worth reporting.

        The decisive field is ``enforced``, which is measured rather than inferred: it
        runs an unscoped count and requires zero rows. A configuration that merely
        looks correct -- policy present, RLS forced -- still reports
        ``enforced: false`` if the connected role can read across tenants.
        """
        conn = await self._connect(register_vector=False)
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT current_user AS role,
                           (SELECT rolsuper     FROM pg_roles WHERE rolname = current_user)
                               AS is_superuser,
                           (SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user)
                               AS bypasses_rls,
                           (SELECT relrowsecurity      FROM pg_class WHERE relname = 'chunks')
                               AS rls_enabled,
                           (SELECT relforcerowsecurity FROM pg_class WHERE relname = 'chunks')
                               AS rls_forced,
                           EXISTS (
                               SELECT 1 FROM pg_policy pol
                               JOIN pg_class c ON c.oid = pol.polrelid
                               WHERE c.relname = 'chunks'
                                 AND pol.polname = 'chunks_tenant_isolation'
                           ) AS policy_present
                    """
                )
                facts = dict(await cur.fetchone() or {})
                # The measurement. No tenant is set on this connection, so a policy
                # that is genuinely in force must yield zero rows.
                await cur.execute("SELECT count(*) AS n FROM chunks")
                row = await cur.fetchone()
                leaked = int(row["n"]) if row else 0
        finally:
            await conn.close()

        facts["unscoped_visible_rows"] = leaked
        facts["enforced"] = bool(
            facts.get("rls_enabled")
            and facts.get("policy_present")
            and not facts.get("is_superuser")
            and not facts.get("bypasses_rls")
            and leaked == 0
        )
        return facts

    async def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        embedder: str,
        tenant_id: str,
        redactions: list[dict[str, int]] | None = None,
    ) -> tuple[int, int]:
        """
        Insert or update chunks by ``(tenant_id, content_hash)``.

        Returns ``(inserted, updated)``. Re-ingesting an unchanged document reports
        zero inserts, which is the signal that the pipeline is genuinely idempotent
        rather than merely not crashing.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}")
        bad = [i for i, e in enumerate(embeddings) if len(e) != self.dimensions]
        if bad:
            raise ValueError(
                f"embeddings at {bad[:3]} have wrong width; expected {self.dimensions}"
            )

        marks = redactions or [{} for _ in chunks]
        inserted = 0
        async with self._session(tenant_id) as cur:
            for chunk, embedding, redaction in zip(chunks, embeddings, marks, strict=True):
                await cur.execute(
                    """
                    INSERT INTO chunks
                        (content_hash, tenant_id, source, page, text,
                         embedding, embedder, redactions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, content_hash) DO UPDATE SET
                        embedding   = EXCLUDED.embedding,
                        embedder    = EXCLUDED.embedder,
                        redactions  = EXCLUDED.redactions,
                        ingested_at = now()
                    RETURNING (xmax = 0) AS was_inserted
                    """,
                    (
                        content_hash(chunk.source, chunk.page, chunk.text),
                        tenant_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        embedding,
                        embedder,
                        Jsonb(redaction),
                    ),
                )
                row = await cur.fetchone()
                if row and row["was_inserted"]:
                    inserted += 1
        return inserted, len(chunks) - inserted

    async def vector_search(
        self, embedding: list[float], *, top_k: int, tenant_id: str
    ) -> list[ScoredChunk]:
        """Nearest neighbours by cosine distance, converted to a similarity score."""
        async with self._session(tenant_id) as cur:
            await cur.execute(
                """
                SELECT source, page, text, 1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE tenant_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, tenant_id, embedding, top_k),
            )
            rows = await cur.fetchall()
        return [
            ScoredChunk(
                chunk=Chunk(
                    text=r["text"],
                    source=r["source"],
                    page=r["page"],
                    score=round(float(r["similarity"]), 4),
                ),
                score=float(r["similarity"]),
                rank=i + 1,
            )
            for i, r in enumerate(rows)
        ]

    async def lexical_search(self, query: str, *, top_k: int, tenant_id: str) -> list[ScoredChunk]:
        """
        Full-text search over the generated tsvector.

        This is the half of hybrid retrieval that vectors are bad at: exact identifiers,
        acronyms and rare terms (``ASEAN DEFA``) where an embedding blurs the very token
        that makes the match correct.

        The query is reduced to OR-joined content terms first -- see
        ``lexical_query_terms`` for why passing the raw question matches nothing.
        """
        reduced = lexical_query_terms(query)
        if not reduced:
            return []
        async with self._session(tenant_id) as cur:
            await cur.execute(
                """
                SELECT source, page, text,
                       ts_rank_cd(text_search, websearch_to_tsquery('english', %s)) AS rank
                FROM chunks
                WHERE tenant_id = %s
                  AND text_search @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (reduced, tenant_id, reduced, top_k),
            )
            rows = await cur.fetchall()
        return [
            ScoredChunk(
                chunk=Chunk(
                    text=r["text"],
                    source=r["source"],
                    page=r["page"],
                    score=round(float(r["rank"]), 4),
                ),
                score=float(r["rank"]),
                rank=i + 1,
            )
            for i, r in enumerate(rows)
        ]

    async def coverage(self, *, tenant_id: str) -> dict[str, list[int]]:
        """
        Which pages of which sources are indexed.

        Exposed because a retrieval metric is only valid if the corpus contains the
        ground truth it is scored against. Without this, a case whose ground-truth
        pages were never ingested scores recall 0 and reports an ingest gap as a
        retriever failure -- the same error class as scoring groundedness against a
        context too short to support the answer.
        """
        async with self._session(tenant_id) as cur:
            await cur.execute(
                """
                SELECT source, array_agg(DISTINCT page ORDER BY page) AS pages
                FROM chunks
                WHERE tenant_id = %s AND page IS NOT NULL
                GROUP BY source
                """,
                (tenant_id,),
            )
            rows = await cur.fetchall()
        return {r["source"]: [int(p) for p in r["pages"]] for r in rows}

    async def count(self, *, tenant_id: str) -> int:
        async with self._session(tenant_id) as cur:
            await cur.execute("SELECT count(*) AS n FROM chunks WHERE tenant_id = %s", (tenant_id,))
            row = await cur.fetchone()
            return int(row["n"]) if row else 0
