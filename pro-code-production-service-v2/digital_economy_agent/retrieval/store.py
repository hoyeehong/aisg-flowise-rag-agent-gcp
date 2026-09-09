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

    async def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        embedder: str,
        tenant_id: str = "default",
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
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
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
            await conn.commit()
        finally:
            await conn.close()
        return inserted, len(chunks) - inserted

    async def vector_search(
        self, embedding: list[float], *, top_k: int, tenant_id: str = "default"
    ) -> list[ScoredChunk]:
        """Nearest neighbours by cosine distance, converted to a similarity score."""
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
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
        finally:
            await conn.close()
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

    async def lexical_search(
        self, query: str, *, top_k: int, tenant_id: str = "default"
    ) -> list[ScoredChunk]:
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
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
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
        finally:
            await conn.close()
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

    async def coverage(self, *, tenant_id: str = "default") -> dict[str, list[int]]:
        """
        Which pages of which sources are indexed.

        Exposed because a retrieval metric is only valid if the corpus contains the
        ground truth it is scored against. Without this, a case whose ground-truth
        pages were never ingested scores recall 0 and reports an ingest gap as a
        retriever failure -- the same error class as scoring groundedness against a
        context too short to support the answer.
        """
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
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
        finally:
            await conn.close()
        return {r["source"]: [int(p) for p in r["pages"]] for r in rows}

    async def count(self, *, tenant_id: str = "default") -> int:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) AS n FROM chunks WHERE tenant_id = %s", (tenant_id,)
                )
                row = await cur.fetchone()
                return int(row["n"]) if row else 0
        finally:
            await conn.close()
