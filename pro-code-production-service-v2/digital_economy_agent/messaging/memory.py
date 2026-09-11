"""
An in-process broker that reproduces the behaviours the consumer has to survive.

Not a mock. It redelivers nacked messages, increments the delivery attempt on each
redelivery, tracks unacknowledged age, and keeps acked messages out of subsequent
pulls -- which is exactly the set of behaviours the retry, dead-letter and
ack-after-commit rules depend on. A mock that returned canned values would let those
rules pass while being wrong.

Not for production: there is no durability, no visibility timeout, and no coordination
between processes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .types import IncomingMessage


@dataclass
class _Pending:
    message: IncomingMessage
    first_seen: float


class InMemorySubscriber:
    """A queue of messages with at-least-once semantics."""

    def __init__(self) -> None:
        self._queue: deque[_Pending] = deque()
        self._in_flight: dict[str, _Pending] = {}
        self.acked: list[IncomingMessage] = []
        self.nacked: list[IncomingMessage] = []
        self._counter = 0
        self._lock = asyncio.Lock()

    # --- production side ---------------------------------------------------

    def publish(
        self,
        payload: Any,
        *,
        message_id: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> str:
        """Enqueue a message. ``payload`` is JSON-encoded unless already bytes."""
        self._counter += 1
        identifier = message_id or f"msg-{self._counter}"
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self._queue.append(
            _Pending(
                message=IncomingMessage(
                    id=identifier,
                    data=data,
                    attributes=attributes or {},
                    delivery_attempt=1,
                    published_at=datetime.now(UTC),
                ),
                first_seen=time.monotonic(),
            )
        )
        return identifier

    # --- consumption side --------------------------------------------------

    async def pull(self, max_messages: int) -> list[IncomingMessage]:
        async with self._lock:
            out: list[IncomingMessage] = []
            while self._queue and len(out) < max_messages:
                pending = self._queue.popleft()
                self._in_flight[pending.message.id] = pending
                out.append(pending.message)
            return out

    async def ack(self, message: IncomingMessage) -> None:
        async with self._lock:
            self._in_flight.pop(message.id, None)
            self.acked.append(message)

    async def nack(self, message: IncomingMessage) -> None:
        async with self._lock:
            pending = self._in_flight.pop(message.id, None)
            self.nacked.append(message)
            if pending is None:
                return
            # Redelivery carries a higher attempt count. This is what lets the consumer
            # recognise a message that will never succeed.
            self._queue.append(
                _Pending(
                    message=IncomingMessage(
                        id=message.id,
                        data=message.data,
                        attributes=message.attributes,
                        delivery_attempt=message.delivery_attempt + 1,
                        published_at=message.published_at,
                    ),
                    first_seen=pending.first_seen,
                )
            )

    async def oldest_unacked_age_seconds(self) -> float | None:
        candidates = [p.first_seen for p in self._queue] + [
            p.first_seen for p in self._in_flight.values()
        ]
        if not candidates:
            return 0.0
        return time.monotonic() - min(candidates)

    @property
    def depth(self) -> int:
        """Messages waiting or in flight."""
        return len(self._queue) + len(self._in_flight)


@dataclass
class InMemoryDeadLetters:
    """Collects dead-lettered messages with the reason each was retired."""

    messages: list[tuple[IncomingMessage, str]] = field(default_factory=list)
    fail_next: bool = False

    async def send(self, message: IncomingMessage, *, reason: str) -> None:
        if self.fail_next:
            # Used to prove the consumer does not ack a message it failed to store.
            self.fail_next = False
            raise RuntimeError("dead-letter sink unavailable")
        self.messages.append((message, reason))

    @property
    def ids(self) -> list[str]:
        return [m.id for m, _ in self.messages]
