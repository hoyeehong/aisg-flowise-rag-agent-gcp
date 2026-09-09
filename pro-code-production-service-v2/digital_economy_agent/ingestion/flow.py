"""
Prefect orchestration for the ingestion pipeline.

Prefect is an *optional* dependency (``pip install '.[ingestion]'``). The API service
never imports this module, so the runtime image does not carry an orchestrator it will
never run. All correctness lives in ``pipeline.py``; this file adds only scheduling,
retries and observability.

Retries are safe here precisely because the pipeline is content-hash keyed: a task that
fails after a partial upsert can be retried whole, and the already-written chunks are
updated rather than duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..retrieval.chunking import ChunkingConfig
from ..retrieval.embeddings import GoogleEmbedder, HashingEmbedder
from ..retrieval.store import PgVectorStore
from ..retrieval.types import Embedder
from .pipeline import IngestionResult, ingest_pdf, read_pdf_pages
from .redaction import PatternRedactor

# Prefect is optional. Under TYPE_CHECKING mypy sees only the real imports, so it type
# checks against the genuine decorators; at runtime the except branch supplies no-op
# shims that keep this module importable (and the pipeline callable) without Prefect.
if TYPE_CHECKING:
    from prefect import flow, task
    from prefect.tasks import exponential_backoff

    PREFECT_AVAILABLE = True
else:
    try:
        from prefect import flow, task
        from prefect.tasks import exponential_backoff

        PREFECT_AVAILABLE = True
    except ImportError:  # pragma: no cover - exercised by the import-guard test
        PREFECT_AVAILABLE = False

        def task(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def flow(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def exponential_backoff(**kwargs):
            return None


def _build_embedder(api_key: str | None, dimensions: int) -> Embedder:
    """Google when a key is present, deterministic hashing otherwise (dry runs, CI)."""
    if api_key:
        return GoogleEmbedder(api_key, dimensions=dimensions)
    return HashingEmbedder(dimensions=dimensions)


@task(
    name="ingest-document",
    retries=3,
    # Transient failures here are embedding-provider rate limits and database blips,
    # both of which clear on a timescale of seconds to a minute.
    retry_delay_seconds=exponential_backoff(backoff_factor=10) if PREFECT_AVAILABLE else None,
    retry_jitter_factor=0.5 if PREFECT_AVAILABLE else None,
    tags=["ingestion"],
)
async def ingest_document_task(
    path: str,
    dsn: str,
    *,
    api_key: str | None = None,
    admin_dsn: str | None = None,
    dimensions: int = 768,
    tenant_id: str = "default",
    chunk_size: int = 1000,
    overlap: int = 200,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """
    Ingest one document. Returns a JSON-serialisable summary for the flow run.

    ``admin_dsn`` is used for DDL, which needs ownership the query role does not have.
    It defaults to ``dsn``, keeping a single-DSN setup working.
    """
    store = PgVectorStore(dsn, dimensions=dimensions)
    await PgVectorStore(admin_dsn or dsn, dimensions=dimensions).ensure_schema()
    result: IngestionResult = await ingest_pdf(
        Path(path),
        store=store,
        embedder=_build_embedder(api_key, dimensions),
        redactor=PatternRedactor(),
        chunking=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
        tenant_id=tenant_id,
        max_pages=max_pages,
    )
    return {
        "source": result.source,
        "tenant_id": result.tenant_id,
        "pages": result.pages,
        "chunks": result.chunks,
        "inserted": result.inserted,
        "updated": result.updated,
        "redactions": result.redactions.counts,
        "redactions_low_confidence": result.redactions.low_confidence,
        "embedder": result.embedder,
        "redactor": result.redactor,
        "chunking": result.chunking,
        "idempotent_rerun": result.idempotent_rerun,
    }


@flow(name="ingest-corpus", log_prints=True)
async def ingest_corpus_flow(
    paths: list[str],
    dsn: str,
    *,
    api_key: str | None = None,
    dimensions: int = 768,
    tenant_id: str = "default",
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """
    Ingest a corpus, one document per task.

    Per-document tasks rather than one task for the corpus: a single unreadable PDF
    then fails and retries alone instead of forcing the whole corpus to be re-embedded,
    which is the expensive part.
    """
    results: list[dict[str, Any]] = []
    for path in paths:
        results.append(
            await ingest_document_task(
                path,
                dsn,
                api_key=api_key,
                dimensions=dimensions,
                tenant_id=tenant_id,
                max_pages=max_pages,
            )
        )
    for summary in results:
        print(
            f"{summary['source']}: {summary['chunks']} chunks "
            f"({summary['inserted']} new, {summary['updated']} updated), "
            f"redactions={summary['redactions']}"
        )
    return results


__all__ = [
    "PREFECT_AVAILABLE",
    "ingest_corpus_flow",
    "ingest_document_task",
    "read_pdf_pages",
]
