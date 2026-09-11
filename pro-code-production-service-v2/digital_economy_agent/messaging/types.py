"""
Broker-neutral message types and protocols.

Kafka and Pub/Sub differ in almost every detail and agree on the two that shape the
code: delivery is **at-least-once**, and ordering is only partial (per-partition in
Kafka, per-ordering-key in Pub/Sub, and absent by default in both). Everything in this
package is built for those two facts rather than for one broker's API.

Declared as protocols for the same reason the gateway, embedder and store are: the
processing semantics -- idempotency, retry, dead-lettering, ack ordering -- are the
part worth testing, and testing them against a real broker would make them slow and
flaky without making them better tested.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class IncomingMessage:
    """One delivery. The same message id may arrive more than once."""

    id: str
    data: bytes
    attributes: Mapping[str, str] = field(default_factory=dict)
    # 1 on first delivery. Brokers count redeliveries, and the count is what makes a
    # poison message detectable: without it a payload that always fails is retried
    # forever, and in a partitioned broker it blocks everything behind it.
    delivery_attempt: int = 1
    published_at: datetime | None = None


class Subscriber(Protocol):
    """A pull-based subscription."""

    async def pull(self, max_messages: int) -> list[IncomingMessage]:
        """Fetch up to ``max_messages``. Returns an empty list when none are ready."""
        ...

    async def ack(self, message: IncomingMessage) -> None:
        """Acknowledge. Called only after the handler's work is durable."""
        ...

    async def nack(self, message: IncomingMessage) -> None:
        """Return the message for redelivery."""
        ...

    async def oldest_unacked_age_seconds(self) -> float | None:
        """
        Age of the oldest unacknowledged message, or None if unavailable.

        This is the consumer's real health metric. Throughput looks fine right up to
        the point the consumer stops keeping up, and only backlog age shows that.
        """
        ...


class DeadLetterSink(Protocol):
    """Where messages go when they cannot be processed."""

    async def send(self, message: IncomingMessage, *, reason: str) -> None: ...


class PermanentMessageError(Exception):
    """
    Raised by a handler for a message that will never succeed.

    Malformed payloads belong here. Retrying them burns the delivery budget and delays
    everything behind them to reach a conclusion that was already available on the
    first attempt.
    """


class TransientMessageError(Exception):
    """
    Raised by a handler for a failure that may succeed later.

    The default for an unrecognised exception is *also* to retry, so this type is for
    clarity rather than control flow: an unexpected error should not be assumed
    permanent and silently discarded.
    """
