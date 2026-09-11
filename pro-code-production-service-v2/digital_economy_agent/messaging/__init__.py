"""Event-driven ingestion: broker-neutral consumption with idempotent effects."""

from .consumer import BatchOutcome, Handler, MessageConsumer
from .documents import DocumentIngestionEvent, DocumentIngestionHandler, DocumentPage
from .memory import InMemoryDeadLetters, InMemorySubscriber
from .types import (
    DeadLetterSink,
    IncomingMessage,
    PermanentMessageError,
    Subscriber,
    TransientMessageError,
)

__all__ = [
    "BatchOutcome",
    "DeadLetterSink",
    "DocumentIngestionEvent",
    "DocumentIngestionHandler",
    "DocumentPage",
    "Handler",
    "InMemoryDeadLetters",
    "InMemorySubscriber",
    "IncomingMessage",
    "MessageConsumer",
    "PermanentMessageError",
    "Subscriber",
    "TransientMessageError",
]
