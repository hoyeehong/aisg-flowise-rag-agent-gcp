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

    async def _connect(self) -> psycopg.AsyncConnection[Any]:
        conn = await psycopg.AsyncConnection.connect(self._dsn, row_factory=dict_row)
        await register_vector_async(conn)
        return conn

    async def ensure_schema(self) -> None:
        """Apply the DDL. Idempotent, so it is safe on every service start."""
        ddl = _SCHEMA.read_text(encoding="utf-8")
        if self.dimensions != 768:
            # Keep the declared column width and the embedder in lockstep; a mismatch
            # would otherwise surface as an opaque insert error much later.
            ddl = ddl.replace("VECTOR(768)", f"VECTOR({self.dimensions})")
        conn = await self._connect()
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
        """
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
                    (query, tenant_id, query, top_k),
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
