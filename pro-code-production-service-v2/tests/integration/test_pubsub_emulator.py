"""
The Pub/Sub adapter against the emulator.

Until this existed, `messaging/pubsub.py` was the only module in the service verified
by nothing but its import guard. The consumer's *rules* were tested against the
in-memory broker, but the mapping performed here -- ack ids, delivery attempts, nack as
a zero-second deadline -- had never run against anything that speaks the real protocol,
and it is the layer where a wrong field name fails silently rather than loudly.

The emulator is not the real service, and two limits are worth stating: it does not
enforce IAM, so no authorisation behaviour is covered here, and it does not implement
`oldest_unacked_message_age`, which Pub/Sub exposes through Cloud Monitoring rather
than the data plane. Everything else these tests touch is the same protocol surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncIterator

import pytest

from digital_economy_agent.messaging import (
    IncomingMessage,
    MessageConsumer,
    PermanentMessageError,
)

EMULATOR = os.getenv("PUBSUB_EMULATOR_HOST", "")
PROJECT = os.getenv("PUBSUB_PROJECT_ID", "test-project")


def _pubsub_available() -> bool:
    if not EMULATOR:
        return False
    try:
        import google.pubsub_v1  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _pubsub_available(),
    reason="needs PUBSUB_EMULATOR_HOST and the events extra (uv sync --extra events)",
)


def _clients_for_test() -> tuple[object, object]:
    """
    Admin clients, built the same way the adapter builds its own.

    No emulator wiring here: the client library reads PUBSUB_EMULATOR_HOST itself and
    builds an insecure channel for the async transport. Arranging that by hand looked
    necessary -- the transport module never mentions the variable -- but the generated
    client does it, and duplicating it only adds a second thing to keep correct.
    """
    from digital_economy_agent.messaging.pubsub import _clients

    sub_cls, pub_cls = _clients()
    return sub_cls(), pub_cls()


class Broker:
    """Topic, dead-letter topic and subscription names for one test."""

    def __init__(self, suffix: str) -> None:
        self.topic = f"projects/{PROJECT}/topics/t-{suffix}"
        self.dead_letter_topic = f"projects/{PROJECT}/topics/dlq-{suffix}"
        self.subscription = f"projects/{PROJECT}/subscriptions/s-{suffix}"
        self.dead_letter_subscription = f"projects/{PROJECT}/subscriptions/dlqs-{suffix}"


@pytest.fixture
async def broker() -> AsyncIterator[Broker]:
    """
    A fresh topic and subscription per test.

    Unique names rather than a shared fixture: Pub/Sub redelivers, so a message one
    test failed to acknowledge would surface in the next one as a phantom delivery.
    """
    sub_client, pub_client = _clients_for_test()
    b = Broker(uuid.uuid4().hex[:12])
    for topic in (b.topic, b.dead_letter_topic):
        await pub_client.create_topic(request={"name": topic})
    await sub_client.create_subscription(
        request={
            "name": b.subscription,
            "topic": b.topic,
            "ack_deadline_seconds": 10,
            # Required for delivery_attempt to be populated at all -- see
            # test_a_subscription_without_a_dead_letter_policy_is_refused.
            "dead_letter_policy": {
                "dead_letter_topic": b.dead_letter_topic,
                "max_delivery_attempts": 5,
            },
        }
    )
    await sub_client.create_subscription(
        request={"name": b.dead_letter_subscription, "topic": b.dead_letter_topic}
    )
    try:
        yield b
    finally:
        # Cleanup must not mask a test failure, so every deletion is best-effort.
        for name in (b.subscription, b.dead_letter_subscription):
            with contextlib.suppress(Exception):
                await sub_client.delete_subscription(request={"subscription": name})
        for topic in (b.topic, b.dead_letter_topic):
            with contextlib.suppress(Exception):
                await pub_client.delete_topic(request={"topic": topic})


async def _publish(topic: str, data: bytes, **attributes: str) -> None:
    _, pub_client = _clients_for_test()
    await pub_client.publish(
        request={"topic": topic, "messages": [{"data": data, "attributes": attributes}]}
    )


async def _pull_until(subscriber: object, *, timeout: float = 5.0) -> list[IncomingMessage]:
    """
    Wait for a delivery.

    One call suffices rather than a poll loop: the adapter pulls with
    ``return_immediately=False``, so the request itself blocks until a message is
    available. The timeout is here to fail the test rather than hang it.
    """
    try:
        return await asyncio.wait_for(subscriber.pull(10), timeout)  # type: ignore[attr-defined]
    except TimeoutError:
        return []


async def _stays_empty(subscriber: object, *, seconds: float = 2.0) -> bool:
    """
    Whether nothing is delivered within ``seconds``.

    Asserting an absence needs a bounded wait, and a bare ``pull() == []`` cannot give
    one: long-polling means the call blocks server-side for up to ninety seconds when
    the subscription is empty. That is the right behaviour in production -- it beats a
    busy loop -- and it is why proving emptiness here has to be done with a timeout
    rather than by inspecting a return value.
    """
    try:
        messages = await asyncio.wait_for(subscriber.pull(10), seconds)  # type: ignore[attr-defined]
    except TimeoutError:
        return True
    return not messages


# --- the mapping ------------------------------------------------------------


async def test_pull_maps_every_field_the_consumer_reads(broker: Broker) -> None:
    """
    IncomingMessage is broker-neutral, so this mapping is the only place the Pub/Sub
    wire format is interpreted. A wrong field name here produces an empty attribute
    dict or a zero attempt count, neither of which raises.
    """
    from digital_economy_agent.messaging.pubsub import PubSubSubscriber

    await _publish(broker.topic, b'{"hello": "world"}', tenant_id="acme", source="d.pdf")
    subscriber = PubSubSubscriber(broker.subscription)
    messages = await _pull_until(subscriber)

    assert len(messages) == 1
    message = messages[0]
    assert message.data == b'{"hello": "world"}'
    assert message.attributes == {"tenant_id": "acme", "source": "d.pdf"}
    assert message.id
    assert message.delivery_attempt >= 1
    assert message.published_at is not None


async def test_ack_stops_redelivery(broker: Broker) -> None:
    """The consumer acks last, so an ack that does not take means duplicate work."""
    from digital_economy_agent.messaging.pubsub import PubSubSubscriber

    await _publish(broker.topic, b"once")
    subscriber = PubSubSubscriber(broker.subscription)
    messages = await _pull_until(subscriber)
    await subscriber.ack(messages[0])

    assert await _stays_empty(subscriber), "an acknowledged message was redelivered"


async def test_nack_redelivers_promptly_and_counts_the_attempt(broker: Broker) -> None:
    """
    Pub/Sub has no nack on the pull API; the adapter sets the ack deadline to zero.
    If that mapping were wrong the message would still return -- after the full
    deadline -- so a passing retry test with a short timeout is the only thing that
    distinguishes "works" from "works in ten seconds".
    """
    from digital_economy_agent.messaging.pubsub import PubSubSubscriber

    await _publish(broker.topic, b"retry-me")
    subscriber = PubSubSubscriber(broker.subscription)
    first = (await _pull_until(subscriber))[0]
    await subscriber.nack(first)

    second = await _pull_until(subscriber, timeout=5.0)
    assert second, "nack did not redeliver; the zero-deadline mapping is wrong"
    assert second[0].id == first.id
    assert second[0].delivery_attempt > first.delivery_attempt
    await subscriber.ack(second[0])


# --- the guard that makes the retry ceiling real ----------------------------


async def test_a_subscription_without_a_dead_letter_policy_is_refused(broker: Broker) -> None:
    """
    Pub/Sub populates delivery_attempt only where a dead-letter policy exists. Without
    one the field is absent, every delivery looks like the first, and
    max_delivery_attempts never fires -- so a poison message is redelivered forever.
    Measured against the emulator: such a subscription reports delivery_attempt=0.
    """
    from digital_economy_agent.messaging.pubsub import PubSubSubscriber

    sub_client, _ = _clients_for_test()
    bare = f"{broker.subscription}-bare"
    await sub_client.create_subscription(  # type: ignore[attr-defined]
        request={"name": bare, "topic": broker.topic, "ack_deadline_seconds": 10}
    )
    try:
        await _publish(broker.topic, b"unbounded")
        with pytest.raises(RuntimeError, match="dead-letter policy"):
            await _pull_until(PubSubSubscriber(bare))
    finally:
        await sub_client.delete_subscription(request={"subscription": bare})  # type: ignore[attr-defined]


async def test_the_guard_can_be_opted_out_of(broker: Broker) -> None:
    """The opt-out exists for a subscription someone accepts unbounded retries on."""
    from digital_economy_agent.messaging.pubsub import PubSubSubscriber

    sub_client, _ = _clients_for_test()
    bare = f"{broker.subscription}-optout"
    await sub_client.create_subscription(  # type: ignore[attr-defined]
        request={"name": bare, "topic": broker.topic, "ack_deadline_seconds": 10}
    )
    try:
        await _publish(broker.topic, b"accepted")
        subscriber = PubSubSubscriber(bare, require_delivery_attempt=False)
        messages = await _pull_until(subscriber)
        assert len(messages) == 1
        # Normalised to 1 rather than left at 0, so the consumer's arithmetic has a
        # first attempt to count from.
        assert messages[0].delivery_attempt == 1
    finally:
        await sub_client.delete_subscription(request={"subscription": bare})  # type: ignore[attr-defined]


# --- dead letters -----------------------------------------------------------


async def test_dead_letters_carry_the_reason(broker: Broker) -> None:
    """
    A dead-letter queue of payloads with no explanation is an archive, not a
    diagnostic, so the reason and the original id travel as attributes.
    """
    from digital_economy_agent.messaging.pubsub import PubSubDeadLetters, PubSubSubscriber

    sink = PubSubDeadLetters(broker.dead_letter_topic)
    await sink.send(
        IncomingMessage(id="orig-1", data=b"bad payload", attributes={"tenant_id": "acme"}),
        reason="unparseable JSON",
    )
    landed = await _pull_until(
        PubSubSubscriber(broker.dead_letter_subscription, require_delivery_attempt=False)
    )
    assert len(landed) == 1
    assert landed[0].data == b"bad payload"
    assert landed[0].attributes["dead_letter_reason"] == "unparseable JSON"
    assert landed[0].attributes["original_message_id"] == "orig-1"
    # The original attributes survive; a reason that replaced them would lose the tenant.
    assert landed[0].attributes["tenant_id"] == "acme"


async def test_consumer_dead_letters_a_permanent_failure_end_to_end(broker: Broker) -> None:
    """
    The whole path against a real broker: pull, fail permanently, dead-letter, ack.

    The ack matters as much as the dead-letter. Setting a message aside and leaving it
    unacknowledged would duplicate it -- once in the dead-letter topic and once on the
    next redelivery -- which is how a poison message becomes an infinite one.
    """
    from digital_economy_agent.messaging.pubsub import PubSubDeadLetters, PubSubSubscriber

    await _publish(broker.topic, b"not json at all", tenant_id="acme")
    subscriber = PubSubSubscriber(broker.subscription)
    sink = PubSubDeadLetters(broker.dead_letter_topic)

    async def handler(message: IncomingMessage) -> None:
        raise PermanentMessageError("payload is not JSON")

    consumer = MessageConsumer(subscriber, handler, dead_letters=sink, max_delivery_attempts=5)
    # run_once pulls with long-polling, so one call is enough; the timeout turns a
    # broken mapping into a failure rather than a hang.
    outcome = await asyncio.wait_for(consumer.run_once(), 10.0)

    assert outcome.dead_lettered == 1, outcome.reasons
    assert await _stays_empty(subscriber), "a dead-lettered message was left unacknowledged"

    landed = await _pull_until(
        PubSubSubscriber(broker.dead_letter_subscription, require_delivery_attempt=False)
    )
    assert len(landed) == 1
    assert "not JSON" in landed[0].attributes["dead_letter_reason"]


# --- the emulator itself ----------------------------------------------------


async def test_the_adapter_actually_points_at_the_emulator() -> None:
    """
    Everything above is only meaningful if these clients really reach the emulator.

    Worth pinning explicitly, because the routing is not visible in this repository: it
    happens inside the generated client, which reads PUBSUB_EMULATOR_HOST and swaps in
    an insecure channel. The transport module never mentions the variable, so reading
    the adapter gives no hint that it works -- and if a library upgrade moved it, every
    test above would start passing against production credentials, or failing for
    reasons that look like adapter bugs.
    """
    from google.cloud import pubsub_v1 as handwritten

    from digital_economy_agent.messaging.pubsub import emulator_host

    assert emulator_host() == EMULATOR

    # Asserted behaviourally, not structurally. `transport._host` still reports the
    # production endpoint as metadata even when an injected channel is talking to the
    # emulator, so checking it proves nothing either way -- it is exactly the kind of
    # assertion that passes while the thing it names is false.
    #
    # Instead: create a topic through the adapter's client, then look for it with the
    # *hand-written* client, which reads PUBSUB_EMULATOR_HOST natively and so is an
    # independent oracle for "is this the emulator". If the adapter had gone to real
    # Pub/Sub, the topic would not be here.
    name = f"projects/{PROJECT}/topics/oracle-{uuid.uuid4().hex[:8]}"
    _, pub_client = _clients_for_test()
    await pub_client.create_topic(request={"name": name})  # type: ignore[attr-defined]
    try:
        oracle = handwritten.PublisherClient()
        listed = {t.name for t in oracle.list_topics(request={"project": f"projects/{PROJECT}"})}
        assert name in listed, (
            "a topic created through the adapter is not in the emulator; the adapter is "
            "talking to something else"
        )
    finally:
        await pub_client.delete_topic(request={"topic": name})  # type: ignore[attr-defined]
