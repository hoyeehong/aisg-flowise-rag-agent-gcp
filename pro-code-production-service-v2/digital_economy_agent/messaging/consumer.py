"""
The processing loop: at-least-once delivery turned into effectively-once effects.

Exactly-once delivery is not available from any broker worth using, and pretending
otherwise is how data gets lost. What is achievable is at-least-once delivery combined
with idempotent effects, which is what the content-hash upsert in the store provides.
This module supplies the other half: acknowledging only after the work is durable,
retrying what may succeed, and removing what never will.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..observability import metrics as obs_metrics
from .types import (
    DeadLetterSink,
    IncomingMessage,
    PermanentMessageError,
    Subscriber,
)

logger = logging.getLogger(__name__)

Handler = Callable[[IncomingMessage], Awaitable[None]]


@dataclass
class BatchOutcome:
    """What one poll-and-process cycle did."""

    processed: int = 0
    retried: int = 0
    dead_lettered: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.processed + self.retried + self.dead_lettered


class MessageConsumer:
    """
    Delivery semantics only. The handler owns what a message *means*.

    Keeping the split sharp is what makes this testable: every rule below is about
    acknowledgement and retry, none of it about documents, so the tests drive real
    redelivery through an in-memory broker rather than mocking the rules they check.
    """

    def __init__(
        self,
        subscriber: Subscriber,
        handler: Handler,
        *,
        dead_letters: DeadLetterSink | None = None,
        max_delivery_attempts: int = 5,
        concurrency: int = 4,
    ) -> None:
        if max_delivery_attempts < 1:
            raise ValueError("max_delivery_attempts must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._subscriber = subscriber
        self._handler = handler
        self._dead_letters = dead_letters
        self._max_attempts = max_delivery_attempts
        # Bounded on purpose. The embedding provider rate-limits, and letting the
        # broker set the fan-out converts a backlog into a wall of 429s -- the gateway
        # would then back off on every one of them, making the backlog worse.
        self._semaphore = asyncio.Semaphore(concurrency)
        self._dead_lettered_total = 0

    async def run_once(self, max_messages: int = 10) -> BatchOutcome:
        """Pull one batch and process it. Returns without waiting when none are ready."""
        await self._report_backlog()
        messages = await self._subscriber.pull(max_messages)
        outcome = BatchOutcome()
        if not messages:
            return outcome

        results = await asyncio.gather(
            *(self._guarded(message) for message in messages), return_exceptions=True
        )
        for message, result in zip(messages, results, strict=True):
            if isinstance(result, BaseException):
                # _guarded is not supposed to raise. If it does, the message has been
                # neither acked nor nacked, so it will be redelivered after the ack
                # deadline -- which is the safe direction to fail.
                logger.exception(
                    "consumer bug while handling message %s; leaving it unacknowledged",
                    message.id,
                    exc_info=result,
                )
                outcome.retried += 1
                outcome.reasons.append(f"{message.id}: consumer error {result!r}")
                continue
            disposition, reason = result
            setattr(outcome, disposition, getattr(outcome, disposition) + 1)
            if reason:
                outcome.reasons.append(f"{message.id}: {reason}")
        return outcome

    async def run_forever(self, *, poll_interval: float = 1.0, max_messages: int = 10) -> None:
        """
        Poll until cancelled.

        An empty pull sleeps; a full one does not, so a backlog is drained at full
        speed rather than one batch per interval.
        """
        while True:
            try:
                outcome = await self.run_once(max_messages)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A broker outage must not end the loop: the pod would restart, fail
                # again, and enter a crashloop that looks like an application fault.
                logger.exception("consumer poll failed; retrying after %ss", poll_interval)
                outcome = BatchOutcome()
            if outcome.total == 0:
                await asyncio.sleep(poll_interval)

    # --- internals ---------------------------------------------------------

    async def _report_backlog(self) -> None:
        try:
            age = await self._subscriber.oldest_unacked_age_seconds()
        except Exception:
            logger.debug("backlog age unavailable", exc_info=True)
            return
        # -1 rather than 0 when unknown: a real zero means "fully caught up", and an
        # alert must be able to tell those apart.
        obs_metrics.CONSUMER_BACKLOG_AGE.set(-1.0 if age is None else age)

    async def _guarded(self, message: IncomingMessage) -> tuple[str, str]:
        async with self._semaphore:
            with obs_metrics.CONSUMER_DURATION.time():
                return await self._handle(message)

    async def _handle(self, message: IncomingMessage) -> tuple[str, str]:
        try:
            await self._handler(message)
        except PermanentMessageError as exc:
            # No retry: the answer will not change. Retrying would spend the whole
            # delivery budget and, on a partitioned broker, delay everything behind it.
            return await self._retire(message, f"permanent: {exc}")
        except Exception as exc:
            if message.delivery_attempt >= self._max_attempts:
                detail = f"{type(exc).__name__}: {exc}"
                return await self._retire(
                    message,
                    f"gave up after {message.delivery_attempt} attempts: {detail}",
                )
            logger.warning(
                "message %s failed on attempt %d/%d, returning it for redelivery: %s",
                message.id,
                message.delivery_attempt,
                self._max_attempts,
                exc,
            )
            await self._subscriber.nack(message)
            obs_metrics.CONSUMER_MESSAGES.labels(outcome="retried").inc()
            return "retried", str(exc)

        # Ack last, and only here. The handler returning means its transaction
        # committed, so this is ack-after-commit. Acking first would turn any crash in
        # the handler into silent data loss.
        await self._subscriber.ack(message)
        obs_metrics.CONSUMER_MESSAGES.labels(outcome="processed").inc()
        return "processed", ""

    async def _retire(self, message: IncomingMessage, reason: str) -> tuple[str, str]:
        """Dead-letter a message, then ack it so it stops being redelivered."""
        if self._dead_letters is None:
            # Without somewhere to put it, acking would discard the payload silently.
            # Redelivery is noisy and visible, which is the better failure.
            logger.error(
                "message %s is undeliverable (%s) and no dead-letter sink is configured; "
                "leaving it for redelivery",
                message.id,
                reason,
            )
            await self._subscriber.nack(message)
            obs_metrics.CONSUMER_MESSAGES.labels(outcome="retried").inc()
            return "retried", f"no dead-letter sink: {reason}"

        try:
            await self._dead_letters.send(message, reason=reason)
        except Exception as exc:
            # The sink failing is not a reason to drop the message. Nack and let the
            # next delivery try again.
            logger.exception("dead-letter sink rejected message %s", message.id)
            await self._subscriber.nack(message)
            obs_metrics.CONSUMER_MESSAGES.labels(outcome="retried").inc()
            return "retried", f"dead-letter sink failed: {exc}"

        await self._subscriber.ack(message)
        self._dead_lettered_total += 1
        obs_metrics.CONSUMER_DEAD_LETTERED.set(self._dead_lettered_total)
        obs_metrics.CONSUMER_MESSAGES.labels(outcome="dead_lettered").inc()
        logger.error("message %s dead-lettered: %s", message.id, reason)
        return "dead_lettered", reason
