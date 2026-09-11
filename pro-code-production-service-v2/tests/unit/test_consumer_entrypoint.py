"""
Consumer entrypoint wiring.

The delivery rules are covered against a real in-memory broker in test_consumer.py.
What is left here is the part that decides whether the process should start at all,
and the HTTP surface Kubernetes and Prometheus depend on -- both of which fail in ways
that look like health rather than like errors.
"""

from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager

from digital_economy_agent.config import Settings
from digital_economy_agent.messaging.__main__ import (
    build_embedder,
    build_probe_app,
    build_subscriber,
)
from digital_economy_agent.retrieval import HashingEmbedder


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"groq_api_key": "k", "model_chain": "primary", "gemini_api_key": ""}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# --- refusing to start ------------------------------------------------------


def test_no_database_is_fatal_with_a_useful_message() -> None:
    """
    Checked before connecting, because psycopg's own failure is unhelpful here: with
    no DSN it falls back to a local Unix socket and reports
    /var/run/postgresql/.s.PGSQL.5432 as unreachable, naming a socket nobody
    configured rather than the setting that is missing. Observed running the
    entrypoint inside the container image.
    """
    import asyncio
    from argparse import Namespace

    from digital_economy_agent.messaging.__main__ import _run

    settings = _settings(postgres_dsn="")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("digital_economy_agent.messaging.__main__.get_settings", lambda: settings)
        with pytest.raises(SystemExit, match="no database"):
            asyncio.run(_run(Namespace(allow_hashing_embedder=True)))


def test_no_embedding_credentials_is_fatal() -> None:
    """
    Falling back silently would write vectors with no semantic recall into the same
    table the API reads, leaving the corpus as two incompatible halves with nothing
    reporting it.
    """
    with pytest.raises(SystemExit, match="no embedding credentials"):
        build_embedder(_settings(), allow_hashing=False)


def test_hashing_embedder_requires_an_explicit_opt_in() -> None:
    embedder = build_embedder(_settings(), allow_hashing=True)
    assert isinstance(embedder, HashingEmbedder)


def test_allow_hashing_embedder_setting_also_opts_in() -> None:
    """CI sets this in the environment rather than passing a flag."""
    embedder = build_embedder(_settings(allow_hashing_embedder=True), allow_hashing=False)
    assert isinstance(embedder, HashingEmbedder)


def test_missing_subscription_is_fatal() -> None:
    """
    A consumer with nothing to attach to must not start. Otherwise the pod goes ready,
    reports healthy, scales on an empty backlog, and consumes nothing -- indefinitely.
    """
    with pytest.raises(SystemExit, match="no subscription"):
        build_subscriber(_settings())


# --- the probe surface ------------------------------------------------------


@pytest.fixture
async def probes():
    state: dict[str, object] = {"running": False}
    app = build_probe_app(_settings(pubsub_subscription="projects/p/subscriptions/s"), state)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, state


async def test_liveness_stays_up_while_the_process_is_up(probes) -> None:
    """
    Restarting cannot fix a broker outage, so liveness must not fail on one. A failing
    liveness probe would turn an upstream outage into a crashloop.
    """
    client, state = probes
    state["running"] = False
    assert (await client.get("/healthz")).status_code == 200


async def test_readiness_reports_503_when_the_loop_is_not_running(probes) -> None:
    """
    A pod whose consumer task died but whose process survived would otherwise stay in
    the Service endpoints looking healthy while draining nothing.
    """
    client, state = probes
    state["running"] = False
    response = await client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["consumer_running"] is False


async def test_readiness_is_200_once_the_loop_is_running(probes) -> None:
    client, state = probes
    state["running"] = True
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


async def test_metrics_are_prometheus_exposition(probes) -> None:
    """Without a scrape target a stalled consumer is indistinguishable from an idle one."""
    client, _ = probes
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP" in response.text


async def test_probe_routes_need_no_credential(probes) -> None:
    """Probes and scrapers have none; demanding one makes every pod unready."""
    client, _ = probes
    for path in ("/healthz", "/readyz", "/metrics"):
        assert (await client.get(path)).status_code != 401
