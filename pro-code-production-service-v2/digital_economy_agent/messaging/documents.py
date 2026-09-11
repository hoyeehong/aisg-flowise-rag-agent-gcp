"""
The handler that turns a document event into ingested chunks.

Separate from the consumer on purpose: the consumer knows about acknowledgement and
retry, this knows about documents, and neither has to be tested through the other.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..ingestion.pipeline import IngestionResult, ingest_pages
from ..ingestion.redaction import Redactor
from ..retrieval.chunking import ChunkingConfig
from ..retrieval.store import PgVectorStore
from ..retrieval.types import Embedder
from ..tenancy import is_valid_tenant_id
from .types import IncomingMessage, PermanentMessageError

logger = logging.getLogger(__name__)


class DocumentPage(BaseModel):
    page: int = Field(ge=1, description="1-based, as cited.")
    text: str


class DocumentIngestionEvent(BaseModel):
    """
    A document to ingest, with its pages inline.

    Inline text rather than a storage pointer keeps the consumer free of an object-store
    dependency, and is what lets these tests run with no cloud at all. It also caps the
    document size at the broker's message limit -- 10 MiB for Pub/Sub, 1 MiB by default
    for Kafka. A larger corpus wants an event carrying a URI plus a fetch step in this
    handler; the consumer above would not change.
    """

    tenant_id: str
    source: str = Field(min_length=1, max_length=512)
    pages: list[DocumentPage] = Field(min_length=1)
    # When the source document last changed, per the publisher. Threaded to the store,
    # where it prevents a redelivered older version overwriting a newer one.
    source_updated_at: datetime | None = None

    @field_validator("tenant_id")
    @classmethod
    def _check_tenant(cls, value: str) -> str:
        if not is_valid_tenant_id(value):
            raise ValueError(f"not a usable tenant id: {value!r}")
        return value


class DocumentIngestionHandler:
    """
    Ingests documents from events.

    **Trust boundary.** The tenant comes from the message body, so whoever can publish
    to the topic can write into any tenant they name. Unlike the HTTP path there is no
    credential to verify here -- the authorisation boundary is the topic's IAM policy,
    and it is coarse: publish access is access to every tenant. Narrowing it means a
    topic per tenant, or signed events carrying a verifiable issuer. Neither is
    implemented, and the README says so rather than leaving it to be discovered.
    """

    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        *,
        redactor: Redactor | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._redactor = redactor
        self._chunking = chunking

    async def __call__(self, message: IncomingMessage) -> None:
        event = self._parse(message)
        result = await self._ingest(event)
        logger.info(
            "message %s ingested %s for %s: %d chunks (%d new, %d updated, %d unchanged, %d stale)",
            message.id,
            event.source,
            event.tenant_id,
            result.chunks,
            result.inserted,
            result.updated,
            result.unchanged,
            result.stale,
        )

    def _parse(self, message: IncomingMessage) -> DocumentIngestionEvent:
        try:
            payload = json.loads(message.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Permanent: a payload that is not JSON will not become JSON on the fourth
            # delivery. Retrying it would spend the delivery budget and, on a
            # partitioned broker, hold up every message behind it.
            raise PermanentMessageError(f"payload is not valid JSON: {exc}") from exc
        try:
            return DocumentIngestionEvent.model_validate(payload)
        except ValidationError as exc:
            raise PermanentMessageError(f"payload is not a document event: {exc}") from exc

    async def _ingest(self, event: DocumentIngestionEvent) -> IngestionResult:
        # Anything raised from here is left to propagate as a transient failure: a
        # database or embedding-provider fault is exactly what redelivery is for.
        return await ingest_pages(
            [(p.page, p.text) for p in event.pages],
            source=event.source,
            store=self._store,
            embedder=self._embedder,
            redactor=self._redactor,
            chunking=self._chunking,
            tenant_id=event.tenant_id,
            source_updated_at=event.source_updated_at,
        )
