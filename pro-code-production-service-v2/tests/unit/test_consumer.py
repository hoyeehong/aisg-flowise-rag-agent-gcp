"""
Delivery semantics of the ingestion consumer.

Every test here drives a real in-memory broker rather than asserting on mocks: nacked
messages genuinely come back, with a higher delivery attempt, which is what the retry
and dead-letter rules are built on. Asserting that `nack` "was called" would let a
broken redelivery path pass.
"""

from __future__ import annotations

import asyncio

import pytest

from digital_economy_agent.messaging import (
    IncomingMessage,
    InMemoryDeadLetters,
    InMemorySubscriber,
    MessageConsumer,
    PermanentMessageError,
)


async def _noop(message: IncomingMessage) -> None:
    return None


def _failing(exc: Exception):
    async def handler(message: IncomingMessage) -> None:
        raise exc

    return handler


# --- the happy path --------------------------------------------------------


async def test_a_handled_message_is_acked_once() -> None:
    sub = InMemorySubscriber()
    sub.publish({"hello": "world"})
    consumer = MessageConsumer(sub, _noop)

    outcome = await consumer.run_once()

    assert (outcome.processed, outcome.retried, outcome.dead_lettered) == (1, 0, 0)
    assert len(sub.acked) == 1
    assert sub.nacked == []
    assert sub.depth == 0, "an acked message must not come back"


async def test_an_empty_pull_does_nothing() -> None:
    consumer = MessageConsumer(InMemorySubscriber(), _noop)
    outcome = await consumer.run_once()
    assert outcome.total == 0


async def test_the_ack_happens_after_the_handler_finishes() -> None:
    """
    Ack-after-commit. The handler returning is what makes its work durable, so acking
    before it returns turns any crash inside the handler into silent data loss.
    """
    sub = InMemorySubscriber()
    sub.publish({"doc": 1})
    observed: list[int] = []

    async def handler(message: IncomingMessage) -> None:
        # Nothing may have been acknowledged while the work is still in progress.
        observed.append(len(sub.acked))

    await MessageConsumer(sub, handler).run_once()
    assert observed == [0], "the message was acknowledged before the handler completed"
    assert len(sub.acked) == 1


# --- retry -----------------------------------------------------------------


async def test_a_failure_is_returned_for_redelivery_with_a_higher_attempt() -> None:
    sub = InMemorySubscriber()
    sub.publish({"doc": 1})
    consumer = MessageConsumer(sub, _failing(RuntimeError("database down")))

    first = await consumer.run_once()
    assert (first.processed, first.retried) == (0, 1)
    assert sub.acked == [], "a failed message must not be acknowledged"

    # It comes back, and the broker says this is the second attempt.
    redelivered = await sub.pull(10)
    assert len(redelivered) == 1
    assert redelivered[0].delivery_attempt == 2


async def test_a_message_that_always_fails_is_eventually_dead_lettered() -> None:
    """
    Bounded retries. Without a ceiling a permanently failing message is redelivered
    forever, and on a partitioned broker it blocks every message behind it.
    """
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters()
    sub.publish({"doc": 1}, message_id="poison")
    consumer = MessageConsumer(
        sub, _failing(RuntimeError("nope")), dead_letters=dlq, max_delivery_attempts=3
    )

    outcomes = [await consumer.run_once() for _ in range(3)]

    assert [o.retried for o in outcomes] == [1, 1, 0]
    assert outcomes[-1].dead_lettered == 1
    assert dlq.ids == ["poison"]
    assert "gave up after 3 attempts" in dlq.messages[0][1]
    assert "RuntimeError: nope" in dlq.messages[0][1]
    # Acked only after it was safely stored elsewhere, so it stops being redelivered.
    assert [m.id for m in sub.acked] == ["poison"]
    assert sub.depth == 0


async def test_a_permanent_error_skips_retrying_entirely() -> None:
    """
    A malformed payload will not parse on the fourth attempt either. Retrying spends
    the delivery budget to reach a conclusion available on the first attempt.
    """
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters()
    sub.publish(b"{not json", message_id="bad")
    consumer = MessageConsumer(
        sub, _failing(PermanentMessageError("payload is not valid JSON")), dead_letters=dlq
    )

    outcome = await consumer.run_once()

    assert outcome.dead_lettered == 1
    assert outcome.retried == 0, "a permanent failure must not be retried"
    assert dlq.ids == ["bad"]
    assert dlq.messages[0][1].startswith("permanent:")
    assert sub.nacked == []


# --- refusing to lose messages --------------------------------------------


async def test_without_a_dead_letter_sink_the_message_is_kept() -> None:
    """
    Acking an undeliverable message with nowhere to put it would discard the payload
    silently. Endless redelivery is noisy and visible, which is the better failure.
    """
    sub = InMemorySubscriber()
    sub.publish({"doc": 1}, message_id="orphan")
    consumer = MessageConsumer(sub, _failing(PermanentMessageError("bad")), max_delivery_attempts=1)

    outcome = await consumer.run_once()

    assert outcome.dead_lettered == 0
    assert outcome.retried == 1
    assert sub.acked == [], "the message was dropped with nowhere to store it"
    assert sub.depth == 1


async def test_a_failing_dead_letter_sink_does_not_lose_the_message() -> None:
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters(fail_next=True)
    sub.publish({"doc": 1}, message_id="dlq-fail")
    consumer = MessageConsumer(
        sub, _failing(PermanentMessageError("bad")), dead_letters=dlq, max_delivery_attempts=1
    )

    first = await consumer.run_once()
    assert first.retried == 1
    assert sub.acked == [], "acked despite the dead-letter write failing"

    # The sink recovers, and the redelivery is retired properly.
    second = await consumer.run_once()
    assert second.dead_lettered == 1
    assert dlq.ids == ["dlq-fail"]


async def test_a_failure_to_ack_leaves_the_message_unacknowledged() -> None:
    """
    A bug in the consumer itself must fail towards redelivery, not towards loss.

    The message has been neither acked nor nacked here, so it stays in flight and the
    broker redelivers it after the ack deadline.
    """
    sub = InMemorySubscriber()
    sub.publish({"doc": 1})

    async def exploding_ack(message: IncomingMessage) -> None:
        raise RuntimeError("ack failed")

    sub.ack = exploding_ack  # type: ignore[method-assign]
    outcome = await MessageConsumer(sub, _noop).run_once()

    assert outcome.retried == 1
    assert any("consumer error" in r for r in outcome.reasons)


# --- isolation between messages -------------------------------------------


async def test_one_poison_message_does_not_block_the_others() -> None:
    """
    Head-of-line blocking is the failure this avoids: in a partitioned broker a single
    unprocessable message stalls its whole partition.
    """
    sub = InMemorySubscriber()
    dlq = InMemoryDeadLetters()
    sub.publish({"ok": 1}, message_id="good-1")
    sub.publish({"bad": True}, message_id="poison")
    sub.publish({"ok": 2}, message_id="good-2")

    async def handler(message: IncomingMessage) -> None:
        if message.id == "poison":
            raise PermanentMessageError("cannot parse")

    outcome = await MessageConsumer(sub, handler, dead_letters=dlq).run_once()

    assert outcome.processed == 2
    assert outcome.dead_lettered == 1
    assert sorted(m.id for m in sub.acked) == ["good-1", "good-2", "poison"]
    assert dlq.ids == ["poison"]


async def test_concurrency_is_bounded() -> None:
    """
    The embedding provider rate-limits, so letting the broker set the fan-out turns a
    backlog into a wall of 429s -- and the gateway then backs off on each one, making
    the backlog worse.
    """
    sub = InMemorySubscriber()
    for i in range(12):
        sub.publish({"doc": i})

    active = 0
    peak = 0

    async def handler(message: IncomingMessage) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    outcome = await MessageConsumer(sub, handler, concurrency=3).run_once(max_messages=12)

    assert outcome.processed == 12
    assert peak <= 3, f"ran {peak} handlers concurrently with a limit of 3"


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_configuration_is_rejected(bad: int) -> None:
    sub = InMemorySubscriber()
    with pytest.raises(ValueError):
        MessageConsumer(sub, _noop, max_delivery_attempts=bad)
    with pytest.raises(ValueError):
        MessageConsumer(sub, _noop, concurrency=bad)


# --- observability ---------------------------------------------------------


async def test_backlog_age_is_reported() -> None:
    """
    Backlog age is the metric worth alerting on: throughput looks healthy right up to
    the moment the consumer stops keeping up.
    """
    from digital_economy_agent.observability import metrics as obs

    sub = InMemorySubscriber()
    sub.publish({"doc": 1})
    consumer = MessageConsumer(sub, _noop)

    # The gauge is sampled before the pull, so it reports the backlog about to be
    # worked on: one just-published message, age near zero but not zero.
    await consumer.run_once()
    pending_age = obs.CONSUMER_BACKLOG_AGE._value.get()
    assert 0.0 <= pending_age < 1.0, pending_age

    # Second cycle with nothing left: a real zero, meaning caught up.
    await consumer.run_once()
    assert obs.CONSUMER_BACKLOG_AGE._value.get() == 0.0


async def test_an_unreported_backlog_is_distinguishable_from_zero() -> None:
    """-1, not 0: an alert must be able to tell "caught up" from "cannot tell"."""
    from digital_economy_agent.observability import metrics as obs

    sub = InMemorySubscriber()
    sub.publish({"doc": 1})

    async def unknown() -> float | None:
        return None

    sub.oldest_unacked_age_seconds = unknown  # type: ignore[method-assign]
    await MessageConsumer(sub, _noop).run_once()
    assert obs.CONSUMER_BACKLOG_AGE._value.get() == -1.0


# --- the polling loop ------------------------------------------------------


async def test_run_forever_survives_a_broker_outage() -> None:
    """
    A broker fault must not end the loop. The pod would restart, fail again, and enter
    a crashloop that reads as an application bug.
    """
    sub = InMemorySubscriber()
    sub.publish({"doc": 1})
    calls = 0
    original = sub.pull

    async def flaky(max_messages: int) -> list[IncomingMessage]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("broker unreachable")
        return await original(max_messages)

    sub.pull = flaky  # type: ignore[method-assign]
    consumer = MessageConsumer(sub, _noop)
    task = asyncio.create_task(consumer.run_forever(poll_interval=0.01))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if sub.acked:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(sub.acked) == 1, "the loop did not recover from the broker error"
    assert calls >= 2
