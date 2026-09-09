"""
Document ingestion: parse -> redact -> chunk -> embed -> upsert.

Deliberately free of any orchestrator import. The pipeline is a plain async function so
it can be unit-tested, called from a script, or wrapped by Prefect (see ``flow.py``)
without the tests needing an orchestrator installed. Orchestration is a deployment
concern, not a correctness one.

Idempotency and retry-safety are properties of the design, not of careful operation:
every chunk is keyed by a content hash of (source, page, text), so re-running over an
unchanged document updates rows in place. A run that dies halfway can simply be
repeated -- already-ingested chunks are updated, the rest are inserted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval.chunking import ChunkingConfig, chunk_pages
from ..retrieval.store import PgVectorStore
from ..retrieval.types import Embedder, EmbedTask
from ..tools.types import Chunk
from .redaction import PatternRedactor, RedactionReport, Redactor

logger = logging.getLogger(__name__)

# Embedding providers are per-item or small-batch; this bounds memory and makes partial
# progress visible in logs on a long corpus.
EMBED_BATCH_SIZE = 16


@dataclass
class IngestionResult:
    """Outcome plus the provenance needed to reproduce or audit the run."""

    source: str
    tenant_id: str
    pages: int = 0
    chunks: int = 0
    inserted: int = 0
    updated: int = 0
    redactions: RedactionReport = field(default_factory=RedactionReport)
    embedder: str = ""
    chunking: dict[str, int] = field(default_factory=dict)
    redactor: str = ""

    @property
    def idempotent_rerun(self) -> bool:
        """True when the run changed nothing new -- the signal ingestion is repeatable."""
        return self.chunks > 0 and self.inserted == 0


def read_pdf_pages(path: Path, *, max_pages: int | None = None) -> list[tuple[int, str]]:
    """Extract ``(page_number, text)`` pairs. Page numbers are 1-based, as cited."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        if max_pages is not None and index > max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a single malformed page must not fail the corpus
            logger.warning("page %d of %s could not be extracted: %s", index, path.name, exc)
            continue
        if text.strip():
            pages.append((index, text))
    return pages


async def ingest_pages(
    pages: list[tuple[int, str]],
    *,
    source: str,
    store: PgVectorStore,
    embedder: Embedder,
    redactor: Redactor | None = None,
    chunking: ChunkingConfig | None = None,
    tenant_id: str = "default",
) -> IngestionResult:
    """
    Run the pipeline over already-extracted pages.

    Redaction happens *before* chunking and embedding, so PII never reaches the
    embedding provider and never lands in the vector column. Redacting after embedding
    would leave the original text recoverable in the vector's neighbourhood and would
    have already sent it to a third party.
    """
    active_redactor = redactor if redactor is not None else PatternRedactor()
    chunk_config = chunking or ChunkingConfig()

    result = IngestionResult(
        source=source,
        tenant_id=tenant_id,
        pages=len(pages),
        embedder=embedder.name,
        redactor=active_redactor.name,
        chunking={"chunk_size": chunk_config.chunk_size, "overlap": chunk_config.overlap},
    )

    redacted_pages: list[tuple[int, str]] = []
    for page_number, text in pages:
        clean, report = active_redactor.redact(text)
        result.redactions.merge(report)
        redacted_pages.append((page_number, clean))

    chunks: list[Chunk] = chunk_pages(redacted_pages, source=source, config=chunk_config)
    result.chunks = len(chunks)
    if not chunks:
        logger.warning("no chunks produced for %s; nothing to upsert", source)
        return result

    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        embeddings.extend(await embedder.embed([c.text for c in batch], task=EmbedTask.DOCUMENT))
        logger.info(
            "embedded %d/%d chunks of %s", min(start + len(batch), len(chunks)), len(chunks), source
        )

    inserted, updated = await store.upsert(
        chunks, embeddings, embedder=embedder.name, tenant_id=tenant_id
    )
    result.inserted, result.updated = inserted, updated
    logger.info(
        "ingested %s: %d chunks (%d new, %d updated), %d redactions",
        source,
        len(chunks),
        inserted,
        updated,
        result.redactions.total,
    )
    return result


async def ingest_pdf(
    path: Path,
    *,
    store: PgVectorStore,
    embedder: Embedder,
    redactor: Redactor | None = None,
    chunking: ChunkingConfig | None = None,
    tenant_id: str = "default",
    max_pages: int | None = None,
) -> IngestionResult:
    """Convenience wrapper: read a PDF, then ingest its pages."""
    if not path.is_file():
        raise FileNotFoundError(f"no such document: {path}")
    pages = read_pdf_pages(path, max_pages=max_pages)
    return await ingest_pages(
        pages,
        source=path.name,
        store=store,
        embedder=embedder,
        redactor=redactor,
        chunking=chunking,
        tenant_id=tenant_id,
    )
