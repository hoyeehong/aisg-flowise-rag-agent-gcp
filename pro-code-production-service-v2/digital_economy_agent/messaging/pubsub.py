"""
Google Pub/Sub adapter.

Pull-based rather than streaming-pull. The streaming client owns its own threads and
callback scheduling, which would put the retry, dead-letter and acknowledgement rules
under someone else's control flow; a pull loop keeps all of that in `MessageConsumer`
where it is tested.

Pub/Sub rather than Kafka for this deployment: the infrastructure is already GCP, so
this needs no cluster, no broker upgrades and no separate authentication story -- the
subscription is an IAM binding. The `Subscriber` protocol is what makes that a
deployment choice rather than an architectural one; a Kafka adapter implements the same
four methods.

Verified against the Pub/Sub emulator, which speaks the real protocol: the field
mapping, acknowledgement, nack-as-zero-deadline, the delivery-attempt guard and
dead-lettering all run in CI on every pull request. Two things the emulator cannot
cover -- IAM, which it does not enforce, and `oldest_unacked_message_age`, which Pub/Sub
exposes through Cloud Monitoring rather than the data plane. Neither has been exercised
against the real service.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .types import IncomingMessage

logger = logging.getLogger(__name__)

_MISSING = (
    "google-cloud-pubsub is not installed. Install the events extra:\n"
    "    uv sync --extra events\n"
    "The service and its HTTP API do not need it; only the event consumer does."
)


def _clients() -> tuple[Any, Any]:
    """
    Import the Pub/Sub clients, or explain how to get them.

    Deferred to call time, like Prefect in the ingestion flow, so that importing this
    module -- which `messaging/__init__` does not do -- never requires the dependency.
    The runtime image should not carry a broker client it will not use.
    """
    try:
        from google.pubsub_v1 import PublisherAsyncClient, SubscriberAsyncClient
    except ImportError as exc:  # pragma: no cover - exercised by the import-guard test
        raise ImportError(_MISSING) from exc
    return SubscriberAsyncClient, PublisherAsyncClient


def emulator_host() -> str | None:
    """
    ``PUBSUB_EMULATOR_HOST`` if set, e.g. ``localhost:8085``.

    The client library already routes to the emulator when this is set -- the generated
    client builds an insecure channel for it, async transport included -- so nothing
    here has to arrange that. It is exposed only so the entrypoint can say which broker
    it is talking to, because "connected" and "connected to the thing you meant" are
    different facts and only one of them is visible in a log by default.
    """
    return os.environ.get("PUBSUB_EMULATOR_HOST") or None


class PubSubSubscriber:
    """
    A Pub/Sub subscription behind the `Subscriber` protocol.

    **Delivery attempts require a dead-letter policy.** Pub/Sub populates
    `delivery_attempt` only on subscriptions that have one configured; without it the
    field is absent and every delivery looks like the first, so `max_delivery_attempts`
    in the consumer would never trigger and a poison message would be redelivered
    forever. The subscription in Terraform sets a dead-letter policy for exactly this
    reason, and this class raises if it sees the field missing rather than silently
    retrying without a ceiling.
    """

    def __init__(self, subscription: str, *, require_delivery_attempt: bool = True) -> None:
        self._subscription = subscription
        self._require_delivery_attempt = require_delivery_attempt
        self._client: Any | None = None
        self._ack_ids: dict[str, str] = {}

    async def _subscriber(self) -> Any:
        if self._client is None:
            subscriber_cls, _ = _clients()
            self._client = subscriber_cls()
        return self._client

    async def pull(self, max_messages: int) -> list[IncomingMessage]:
        client = await self._subscriber()
        response = await client.pull(
            request={
                "subscription": self._subscription,
                "max_messages": max_messages,
                # Never block the poll loop inside the client: the consumer's own
                # poll_interval decides how long to wait between empty pulls, and a
                # blocking pull here would make that interval unobservable.
                "return_immediately": False,
            }
        )
        messages: list[IncomingMessage] = []
        for received in response.received_messages:
            attempt = getattr(received, "delivery_attempt", 0) or 0
            if attempt < 1:
                if self._require_delivery_attempt:
                    raise RuntimeError(
                        f"subscription {self._subscription} does not report "
                        "delivery_attempt, which means it has no dead-letter policy. "
                        "Without it a permanently failing message is redelivered "
                        "forever. Configure a dead-letter policy, or construct this "
                        "subscriber with require_delivery_attempt=False to accept "
                        "unbounded retries."
                    )
                attempt = 1
            # The ack id is per delivery, not per message, so it is kept here rather
            # than on IncomingMessage -- which stays broker-neutral.
            self._ack_ids[received.message.message_id] = received.ack_id
            messages.append(
                IncomingMessage(
                    id=received.message.message_id,
                    data=received.message.data,
                    attributes=dict(received.message.attributes),
                    delivery_attempt=attempt,
                    published_at=received.message.publish_time,
                )
            )
        return messages

    async def ack(self, message: IncomingMessage) -> None:
        ack_id = self._ack_ids.pop(message.id, None)
        if ack_id is None:
            logger.warning("no ack id for message %s; it will be redelivered", message.id)
            return
        client = await self._subscriber()
        await client.acknowledge(request={"subscription": self._subscription, "ack_ids": [ack_id]})

    async def nack(self, message: IncomingMessage) -> None:
        """
        Return the message immediately by setting its ack deadline to zero.

        Pub/Sub has no explicit nack on the pull API; a zero deadline is the documented
        equivalent and is what makes redelivery prompt rather than waiting out the
        remaining deadline.
        """
        ack_id = self._ack_ids.pop(message.id, None)
        if ack_id is None:
            return
        client = await self._subscriber()
        await client.modify_ack_deadline(
            request={
                "subscription": self._subscription,
                "ack_ids": [ack_id],
                "ack_deadline_seconds": 0,
            }
        )

    async def oldest_unacked_age_seconds(self) -> float | None:
        """
        Not available from the subscriber API.

        Pub/Sub publishes `subscription/oldest_unacked_message_age` through Cloud
        Monitoring, not through the data plane, so returning None here is honest: the
        consumer records -1, which a dashboard can distinguish from a real zero, and
        the alert is defined against the Cloud Monitoring metric instead.
        """
        return None


class PubSubDeadLetters:
    """
    Publishes undeliverable messages to a dead-letter topic.

    Separate from Pub/Sub's own dead-letter policy on purpose, and complementary to it.
    The subscription's policy handles messages the consumer never managed to process;
    this handles the ones it processed and *decided* were undeliverable -- a malformed
    payload, an invalid tenant -- which the broker cannot recognise. The reason travels
    as an attribute, because a dead-letter queue full of payloads with no explanation
    is an archive, not a diagnostic.
    """

    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._client: Any | None = None

    async def _publisher(self) -> Any:
        if self._client is None:
            _, publisher_cls = _clients()
            self._client = publisher_cls()
        return self._client

    async def send(self, message: IncomingMessage, *, reason: str) -> None:
        client = await self._publisher()
        await client.publish(
            request={
                "topic": self._topic,
                "messages": [
                    {
                        "data": message.data,
                        "attributes": {
                            **dict(message.attributes),
                            "dead_letter_reason": reason[:1024],
                            "original_message_id": message.id,
                            "delivery_attempts": str(message.delivery_attempt),
                        },
                    }
                ],
            }
        )
