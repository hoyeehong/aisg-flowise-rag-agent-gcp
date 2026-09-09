"""
Ingestion CLI.

    uv run python -m digital_economy_agent.ingestion --help

A command rather than only a Prefect flow: ingestion needs to be runnable by hand for a
one-off backfill or a local smoke test, without standing up an orchestrator. The flow in
``flow.py`` adds scheduling and retries on top of the same pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ..config import get_settings
from ..retrieval import ChunkingConfig, GoogleEmbedder, HashingEmbedder, PgVectorStore
from ..retrieval.types import Embedder
from .pipeline import ingest_pdf
from .redaction import NullRedactor, PatternRedactor


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    dsn = args.dsn or settings.postgres_dsn
    if not dsn:
        print("[FATAL] no DSN: pass --dsn or set AGENT_POSTGRES_DSN", file=sys.stderr)
        return 2

    api_key = args.api_key or settings.gemini_api_key.get_secret_value()
    embedder: Embedder
    if api_key:
        embedder = GoogleEmbedder(api_key, dimensions=settings.embedding_dimensions)
    elif args.allow_hashing_embedder:
        embedder = HashingEmbedder(dimensions=settings.embedding_dimensions)
        print("[WARN] using the hashing embedder: retrieval will have no semantic recall")
    else:
        print(
            "[FATAL] no embedding credentials. Set GEMINI_API_KEY, or pass\n"
            "        --allow-hashing-embedder to ingest with lexical-only vectors.",
            file=sys.stderr,
        )
        return 2

    store = PgVectorStore(dsn, dimensions=settings.embedding_dimensions)
    await store.ensure_schema()

    exit_code = 0
    for raw in args.paths:
        path = Path(raw)
        try:
            result = await ingest_pdf(
                path,
                store=store,
                embedder=embedder,
                redactor=NullRedactor() if args.no_redaction else PatternRedactor(),
                chunking=ChunkingConfig(chunk_size=args.chunk_size, overlap=args.overlap),
                tenant_id=args.tenant,
                max_pages=args.max_pages,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        print(
            f"{result.source}: {result.pages} pages -> {result.chunks} chunks "
            f"({result.inserted} new, {result.updated} updated)"
        )
        print(f"  embedder   : {result.embedder}")
        print(f"  chunking   : {result.chunking}")
        print(f"  redactor   : {result.redactor}")
        print(f"  redactions : {result.redactions.counts or 'none'}")
        if result.redactions.low_confidence:
            print(f"  low-conf   : {result.redactions.low_confidence} (redacted, checksum failed)")
        if result.idempotent_rerun:
            print("  rerun      : idempotent (nothing new inserted)")
    print(f"total rows in tenant {args.tenant!r}: {await store.count(tenant_id=args.tenant)}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m digital_economy_agent.ingestion",
        description="Ingest documents into the pgvector store (idempotent).",
    )
    parser.add_argument("paths", nargs="+", help="PDF paths to ingest")
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: AGENT_POSTGRES_DSN)")
    parser.add_argument("--tenant", default="default", help="tenant id to ingest into")
    parser.add_argument("--api-key", default=None, help="Google API key for embeddings")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None, help="stop after N pages")
    parser.add_argument(
        "--allow-hashing-embedder",
        action="store_true",
        help="ingest without an API key using deterministic lexical vectors (dev only)",
    )
    parser.add_argument(
        "--no-redaction",
        action="store_true",
        help="disable PII redaction (only for corpora already cleared for storage)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
