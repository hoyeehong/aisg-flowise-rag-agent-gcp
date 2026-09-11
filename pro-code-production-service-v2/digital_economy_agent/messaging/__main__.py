"""
Consumer entrypoint.

    uv run python -m digital_economy_agent.messaging

A long-running process, not a command that finishes: it attaches to a subscription and
drains it until told to stop. Run as its own workload rather than a thread inside the
API, because the two scale on different signals -- the API on request latency, this on
subscription backlog -- and a burst of ingestion should not slow report generation or
be throttled by an HPA watching request rate.

The process also serves ``/healthz``, ``/readyz`` and ``/metrics``. Without an HTTP
surface a Kubernetes Deployment has no probe target and Prometheus has nothing to
scrape, so a consumer that had silently stopped pulling would look exactly like one
with no work to do.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, Response

from ..config import Settings, get_settings
from ..observability import metrics as obs_metrics
from ..retrieval import GoogleEmbedder, HashingEmbedder, PgVectorStore
from ..retrieval.types import Embedder
from .consumer import MessageConsumer
from .documents import DocumentIngestionHandler
from .types import DeadLetterSink, Subscriber

logger = logging.getLogger(__name__)


def build_embedder(settings: Settings, *, allow_hashing: bool) -> Embedder:
    """
    Pick the embedder, refusing to guess.

    A consumer that silently fell back to the hashing embedder would write vectors with
    no semantic recall into the same table the API reads, and nothing downstream would
    report that the corpus had become two incompatible halves.
    """
    api_key = settings.gemini_api_key.get_secret_value()
    if api_key:
        return GoogleEmbedder(api_key, dimensions=settings.embedding_dimensions)
    if allow_hashing or settings.allow_hashing_embedder:
        logger.warning(
            "using the hashing embedder: ingested chunks will have no semantic recall, "
            "and will not be comparable with chunks embedded by a real model"
        )
        return HashingEmbedder(dimensions=settings.embedding_dimensions)
    raise SystemExit(
        "no embedding credentials: set GEMINI_API_KEY, or pass --allow-hashing-embedder "
        "(local and CI only) to run with lexical-only vectors"
    )


def build_subscriber(settings: Settings) -> tuple[Subscriber, DeadLetterSink | None]:
    """Attach to Pub/Sub. Raises rather than starting with nothing to consume."""
    if not settings.pubsub_subscription:
        raise SystemExit(
            "no subscription: set AGENT_PUBSUB_SUBSCRIPTION to a full resource name, "
            "e.g. projects/<project>/subscriptions/<name>"
        )
    # Imported here, not at module scope: google-cloud-pubsub is an optional extra, and
    # importing it eagerly would make the whole package require it.
    from .pubsub import PubSubDeadLetters, PubSubSubscriber, emulator_host

    host = emulator_host()
    if host:
        # Said out loud because the routing is invisible otherwise: the client library
        # silently redirects to the emulator, so a process that looks like it is
        # draining production is draining a local fake, and the reverse mistake --
        # expecting a fake and getting production -- is the expensive one.
        logger.warning("PUBSUB_EMULATOR_HOST=%s -- consuming from the emulator", host)

    subscriber = PubSubSubscriber(settings.pubsub_subscription)
    sink: DeadLetterSink | None = None
    if settings.pubsub_dead_letter_topic:
        sink = PubSubDeadLetters(settings.pubsub_dead_letter_topic)
    else:
        logger.warning(
            "no dead-letter topic configured: a message that exhausts its retries will "
            "be left unacknowledged and redelivered indefinitely rather than set aside"
        )
    return subscriber, sink


def build_probe_app(settings: Settings, state: dict[str, Any]) -> FastAPI:
    """
    The consumer's HTTP surface: liveness, readiness, metrics.

    Readiness reports whether the poll loop is actually running. A pod whose consumer
    task has died but whose process is alive would otherwise stay in the Service's
    endpoints and look healthy while draining nothing.
    """
    app = FastAPI(title="digital-economy-agent consumer", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        # Liveness stays up while the process is up: restarting cannot fix a broker
        # outage, and a restart loop would only slow recovery.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, object]:
        running = bool(state.get("running"))
        if not running:
            response.status_code = 503
        return {
            "status": "ready" if running else "degraded",
            "checks": {"consumer_running": running},
            "subscription": settings.pubsub_subscription,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=obs_metrics.render(), media_type=obs_metrics.CONTENT_TYPE)

    return app


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    if not settings.postgres_dsn:
        # Checked before connecting. Without it psycopg falls back to a local Unix
        # socket and the failure arrives as an OperationalError naming
        # /var/run/postgresql/.s.PGSQL.5432 -- which describes a socket nobody
        # configured rather than the setting that is missing.
        raise SystemExit("no database: set AGENT_POSTGRES_DSN")

    # DDL and role management need ownership the query role does not have; the admin
    # DSN falls back to the app DSN when none is configured.
    admin = PgVectorStore(settings.admin_dsn, dimensions=settings.embedding_dimensions)
    await admin.ensure_schema()
    if settings.postgres_app_role:
        await admin.ensure_app_role(
            settings.postgres_app_role,
            password=settings.postgres_app_password.get_secret_value() or None,
        )

    store = PgVectorStore(settings.postgres_dsn, dimensions=settings.embedding_dimensions)
    isolation = await store.isolation_status()
    if not isolation["enforced"]:
        # Not fatal: a single-tenant deployment may legitimately run as the owner. It
        # is logged loudly because the consumer writes whatever tenant a message names,
        # so an unenforced policy means a message can write anywhere.
        logger.warning(
            "tenant isolation is NOT enforced on this connection (role=%s superuser=%s "
            "bypassrls=%s); a message naming any tenant will be written to it",
            isolation.get("role"),
            isolation.get("is_superuser"),
            isolation.get("bypasses_rls"),
        )

    handler = DocumentIngestionHandler(
        store, build_embedder(settings, allow_hashing=args.allow_hashing_embedder)
    )
    subscriber, dead_letters = build_subscriber(settings)
    consumer = MessageConsumer(
        subscriber,
        handler,
        dead_letters=dead_letters,
        max_delivery_attempts=settings.consumer_max_delivery_attempts,
        concurrency=settings.consumer_concurrency,
    )

    state: dict[str, Any] = {"running": False}
    server = uvicorn.Server(
        uvicorn.Config(
            build_probe_app(settings, state),
            # Binds all interfaces because it is a container port reached through the
            # Service, not a host-exposed one.
            host="0.0.0.0",
            port=settings.consumer_port,
            log_level="warning",
        )
    )

    async def drain() -> None:
        state["running"] = True
        try:
            await consumer.run_forever(
                poll_interval=settings.consumer_poll_interval_seconds,
                max_messages=settings.consumer_max_messages,
            )
        finally:
            state["running"] = False

    consumer_task: asyncio.Task[Any] = asyncio.create_task(drain(), name="consumer")
    server_task: asyncio.Task[Any] = asyncio.create_task(server.serve(), name="probes")

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # SIGTERM is what Kubernetes sends before the grace period. Handling it lets an
        # in-flight batch finish and acknowledge; without this the pod is killed
        # mid-handler and every message it held is redelivered.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stopping.set)

    logger.info(
        "consumer started: subscription=%s concurrency=%d max_attempts=%d probes on :%d",
        settings.pubsub_subscription,
        settings.consumer_concurrency,
        settings.consumer_max_delivery_attempts,
        settings.consumer_port,
    )

    stop_task: asyncio.Task[Any] = asyncio.create_task(stopping.wait(), name="signal")
    watched: list[asyncio.Task[Any]] = [consumer_task, server_task, stop_task]
    done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
    stop_task.cancel()

    server.should_exit = True
    consumer_task.cancel()
    for task in (consumer_task, server_task):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # A consumer task that finished on its own ended abnormally: run_forever only
    # returns by cancellation. Exit non-zero so the pod restarts rather than sitting
    # there having quietly stopped consuming.
    for task in done:
        if task is consumer_task and not task.cancelled():
            logger.error("consumer loop exited unexpectedly")
            return 1
    logger.info("consumer stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m digital_economy_agent.messaging",
        description="Consume document ingestion events until stopped.",
    )
    parser.add_argument(
        "--allow-hashing-embedder",
        action="store_true",
        help="run without embedding credentials, writing lexical-only vectors",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
