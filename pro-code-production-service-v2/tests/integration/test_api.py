"""
API tests.

These drive the real routes through the real graph, with only the model *provider*
scripted. Mocking the routes or the graph would leave the interesting part -- the
HITL pause spanning two HTTP requests -- untested.
"""

from __future__ import annotations

import json

import httpx
import pytest
from asgi_lifespan import LifespanManager

from digital_economy_agent.api import create_app
from digital_economy_agent.config import Settings
from digital_economy_agent.gateway import ModelGateway, ModelSpec, RetryPolicy

from ..conftest import FakeProvider


@pytest.fixture
def settings() -> Settings:
    # An in-process harness: the in-memory retriever and saver are the intent here, so
    # the durability check is switched off rather than left to fail.
    return Settings(
        groq_api_key="test-key",
        model_chain="primary,secondary",
        max_revisions=2,
        default_top_k=3,
        use_in_memory_retriever=True,
        use_postgres_checkpointer=False,
    )


@pytest.fixture
async def client(settings, retriever, no_sleep):
    provider = FakeProvider("groq")
    gateway = ModelGateway(
        [
            ModelSpec(
                provider="groq", model=m, usd_per_million_input=0.1, usd_per_million_output=0.5
            )
            for m in ("primary", "secondary")
        ],
        {"groq": provider},
        policy=RetryPolicy(),
        sleep=no_sleep,
    )
    app = create_app(settings, gateway=gateway, retriever=retriever)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_healthz_is_dependency_free(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_liveness_stays_ok_while_readiness_drains(retriever, no_sleep):
    """
    The two probes must disagree, and the status codes are what carry that.

    A failing liveness probe triggers a *restart*, which cannot conjure a missing
    credential or database. A failing readiness probe *drains* traffic, which is the
    correct response. So /healthz stays 200 while /readyz answers 503.
    """
    cfg = Settings(
        groq_api_key="",  # missing credential: unservable, but the process is fine
        model_chain="primary",
        use_in_memory_retriever=True,
        use_postgres_checkpointer=False,
    )
    gateway = ModelGateway(
        [ModelSpec(provider="groq", model="primary")],
        {"groq": FakeProvider("groq")},
        sleep=no_sleep,
    )
    app = create_app(cfg, gateway=gateway, retriever=retriever)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            live = await c.get("/healthz")
            ready = await c.get("/readyz")

    assert live.status_code == 200, "liveness must not fail on a missing dependency"
    assert ready.status_code == 503, "readiness must drain traffic"


async def test_readyz_reports_each_check(client):
    r = await client.get("/readyz")
    assert r.status_code == 200, "a ready instance must answer 200 so probes route to it"
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "graph_compiled": True,
        "retriever_ready": True,
        "model_credentials": True,
    }


async def test_readyz_reports_durability_when_it_is_required(retriever, no_sleep):
    """
    With the Postgres checkpointer requested but no DSN, readiness must degrade.

    A service that has been asked for durable human-review pauses and cannot provide
    them is not ready: a restart would silently discard work awaiting a reviewer.
    """
    cfg = Settings(
        groq_api_key="k",
        model_chain="primary",
        use_in_memory_retriever=True,
        use_postgres_checkpointer=True,
        postgres_dsn="",
    )
    gateway = ModelGateway(
        [ModelSpec(provider="groq", model="primary")],
        {"groq": FakeProvider("groq")},
        sleep=no_sleep,
    )
    app = create_app(cfg, gateway=gateway, retriever=retriever)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["durable_checkpointer"] is False
    assert "durable_checkpointer" in body["detail"]


async def test_readyz_asserts_retrieval_deps_only_when_pgvector_selected(retriever, no_sleep):
    """The embedding/Postgres checks must not fire when in-memory retrieval is chosen."""
    cfg = Settings(
        groq_api_key="k",
        model_chain="primary",
        use_in_memory_retriever=True,
        use_postgres_checkpointer=False,
    )
    gateway = ModelGateway(
        [ModelSpec(provider="groq", model="primary")],
        {"groq": FakeProvider("groq")},
        sleep=no_sleep,
    )
    app = create_app(cfg, gateway=gateway, retriever=retriever)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/readyz")

    assert response.status_code == 200, "irrelevant checks must not degrade readiness"
    checks = response.json()["checks"]
    assert "embeddings_configured" not in checks
    assert "postgres_configured" not in checks


async def test_readyz_degrades_without_credentials(retriever, no_sleep):
    """A missing credential must fail readiness, not liveness -- no restart loop."""
    cfg = Settings(groq_api_key="", model_chain="primary")
    gateway = ModelGateway(
        [ModelSpec(provider="groq", model="primary")],
        {"groq": FakeProvider("groq")},
        sleep=no_sleep,
    )
    app = create_app(cfg, gateway=gateway, retriever=retriever)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert (await c.get("/healthz")).status_code == 200
            response = await c.get("/readyz")

    assert response.status_code == 503, "a probe reads the status code, not the body"
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["model_credentials"] is False
    assert "model_credentials" in body["detail"]


async def test_create_report_returns_202_awaiting_review(client):
    """The run pauses for a human, so 202 (accepted, not complete) is the honest code."""
    r = await client.post("/v1/reports", json={"question": "Summarise digital trust in SEA"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "awaiting_review"
    assert body["draft"]
    assert body["final_report"] == ""
    assert body["citations"]
    assert body["prompt_versions"]["research"] == "research.v1"
    assert body["cost"]["llm_calls"] == 2
    assert body["cost"]["total_cost_usd"] > 0


async def test_approve_completes_the_run(client):
    created = (await client.post("/v1/reports", json={"question": "Digital trust in SEA"})).json()
    tid = created["thread_id"]

    r = await client.post(f"/v1/reports/{tid}/review", json={"action": "approve"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["final_report"] == created["draft"]


async def test_revise_then_approve_across_requests(client):
    """The pause must survive between HTTP calls -- that is the point of the checkpointer."""
    tid = (await client.post("/v1/reports", json={"question": "Digital talent gaps"})).json()[
        "thread_id"
    ]

    revised = (
        await client.post(
            f"/v1/reports/{tid}/review",
            json={"action": "revise", "feedback": "Add Vietnam specifics."},
        )
    ).json()
    assert revised["status"] == "awaiting_review"
    assert revised["revisions_used"] == 1
    assert revised["feedback_history"] == ["Add Vietnam specifics."]

    final = (await client.post(f"/v1/reports/{tid}/review", json={"action": "approve"})).json()
    assert final["status"] == "completed"
    assert final["revisions_used"] == 1


async def test_revision_budget_exhaustion_halts(client):
    tid = (
        await client.post(
            "/v1/reports", json={"question": "Digital trust in SEA", "max_revisions": 1}
        )
    ).json()["thread_id"]

    body = (
        await client.post(
            f"/v1/reports/{tid}/review", json={"action": "revise", "feedback": "More."}
        )
    ).json()

    assert body["status"] == "halted"
    assert "revision budget exhausted" in body["halted_reason"]
    assert body["final_report"], "the latest draft is still returned"


async def test_review_on_finished_run_conflicts(client):
    tid = (await client.post("/v1/reports", json={"question": "Digital trust"})).json()["thread_id"]
    await client.post(f"/v1/reports/{tid}/review", json={"action": "approve"})

    r = await client.post(f"/v1/reports/{tid}/review", json={"action": "approve"})
    assert r.status_code == 409
    assert "not awaiting review" in r.json()["detail"]


async def test_revise_without_feedback_is_422(client):
    tid = (await client.post("/v1/reports", json={"question": "Digital trust"})).json()["thread_id"]
    r = await client.post(f"/v1/reports/{tid}/review", json={"action": "revise", "feedback": ""})
    assert r.status_code == 422
    assert "feedback is required" in r.text


async def test_unknown_thread_is_404(client):
    assert (await client.get("/v1/reports/does-not-exist")).status_code == 404
    r = await client.post("/v1/reports/does-not-exist/review", json={"action": "approve"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},  # question missing
        {"question": "hi"},  # below min_length
        {"question": "valid question", "top_k": 0},  # below ge
        {"question": "valid question", "top_k": 999},  # above le
        {"question": "valid question", "junk": 1},  # extra forbidden
    ],
)
async def test_invalid_create_payloads_are_422(client, payload):
    assert (await client.post("/v1/reports", json=payload)).status_code == 422


async def test_metrics_are_prometheus_exposition_not_json(client):
    """
    Phase 1 served JSON here. Readable, but unscrapable: no Prometheus, Cloud
    Monitoring or Grafana could consume it, so nothing could alert on cost or latency.
    """
    await client.post("/v1/reports", json={"question": "Digital trust in SEA"})
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    # Exposition format carries HELP and TYPE lines; JSON does not.
    assert "# HELP agent_llm_cost_usd_total" in body
    assert "# TYPE agent_llm_calls_total counter" in body


def _sample(body: str, metric: str, default: float | None = None, **labels: str) -> float:
    """
    Read one sample out of Prometheus exposition text.

    ``default`` covers a series that does not exist yet, which is what a counter looks
    like before its first increment.
    """
    # prometheus_client emits labels in alphabetical order, not call order.
    label_part = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    needle = f"{metric}{{{label_part}}}" if label_part else metric
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if name.strip() == needle:
            return float(value)
    if default is not None:
        return default
    raise AssertionError(f"{needle} not found in:\n{body[:2000]}")


async def test_metrics_meter_cost_and_tokens_per_model_and_node(client):
    """
    The operational question is which *step* is spending, not just which model.

    Asserted as deltas across one request. Prometheus counters are process-global and
    cumulative by design -- they have no public reset -- so an absolute assertion would
    pass alone and fail in a suite. Reading an increase over a window is also how an
    operator actually consumes a counter.
    """
    before = (await client.get("/metrics")).text
    await client.post("/v1/reports", json={"question": "Digital trust in SEA"})
    after = (await client.get("/metrics")).text

    def delta(metric: str, **labels: str) -> float:
        return _sample(after, metric, default=0.0, **labels) - _sample(
            before, metric, default=0.0, **labels
        )

    assert delta("agent_llm_calls_total", model="groq/primary", node="research") == 1.0
    assert delta("agent_llm_calls_total", model="groq/primary", node="write_draft") == 1.0
    assert delta("agent_llm_tokens_total", model="groq/primary", direction="prompt") == 200.0
    assert delta("agent_llm_tokens_total", model="groq/primary", direction="completion") == 100.0
    assert delta("agent_llm_cost_usd_total", model="groq/primary") > 0
    assert delta("agent_review_pauses_total") == 1.0
    assert delta("agent_retrieval_empty_total") == 0.0, "the dev corpus should match"


async def test_metrics_label_requests_by_route_template_not_path(client):
    """
    Labelling by raw path would create one time series per thread id, which is how a
    metrics backend gets taken down.
    """
    created = (await client.post("/v1/reports", json={"question": "Digital trust"})).json()
    await client.get(f"/v1/reports/{created['thread_id']}")
    body = (await client.get("/metrics")).text

    assert (
        _sample(
            body,
            "agent_requests_total",
            default=0.0,
            route="/v1/reports/{thread_id}",
            status_class="2xx",
        )
        >= 1.0
    ), "the route template must appear as a label"
    assert created["thread_id"] not in body, "a thread id must never become a label"


async def test_stream_emits_node_progress_then_final_state(client):
    """SSE exists because research + drafting take tens of seconds."""
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "/v1/reports/stream", json={"question": "Digital trust in SEA"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: ") and name:
                events.append((name, json.loads(line.removeprefix("data: "))))

    names = [n for n, _ in events]
    assert names[0] == "started"
    assert "node" in names
    assert names[-1] == "state"

    nodes = [d["node"] for n, d in events if n == "node"]
    assert nodes[:2] == ["research", "write_draft"]

    final = events[-1][1]
    assert final["status"] == "awaiting_review"
    assert final["draft"]


# ---------------------------------------------------------------------------
# Provider-failure paths.
#
# These exist because an end-to-end container run against a real Groq endpoint with an
# invalid key returned a bare 500 "Internal Server Error": every earlier test used a
# scripted provider that never raised AllModelsFailed through the API layer. A dependency
# failure must not be reported as a defect in this service.
# ---------------------------------------------------------------------------


async def _client_with_script(settings, retriever, no_sleep, script):
    provider = FakeProvider("groq", script)
    gateway = ModelGateway(
        [ModelSpec(provider="groq", model=m) for m in ("primary", "secondary")],
        {"groq": provider},
        policy=RetryPolicy(max_attempts_per_model=1, cooldown_budget_seconds=0.0),
        sleep=no_sleep,
    )
    app = create_app(settings, gateway=gateway, retriever=retriever)
    return app


async def test_invalid_credentials_return_502_not_500(settings, retriever, no_sleep):
    """A bad API key is an upstream misconfiguration: 502, with a structured body."""
    from digital_economy_agent.gateway import ModelUnavailable

    app = await _client_with_script(
        settings,
        retriever,
        no_sleep,
        {
            "primary": [ModelUnavailable("primary: HTTP 401: Invalid API Key")],
            "secondary": [ModelUnavailable("secondary: HTTP 401: Invalid API Key")],
        },
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/reports", json={"question": "Digital trust in SEA"})

    assert r.status_code == 502
    body = r.json()
    assert body["error"] == "model_gateway_unavailable"
    assert body["detail"]
    # The provider's raw message must not leak to the caller.
    assert "Invalid API Key" not in r.text


async def test_total_rate_limit_returns_503_with_retry_after(settings, retriever, no_sleep):
    """Pure quota exhaustion is retryable, so 503 + Retry-After rather than 502."""
    from digital_economy_agent.gateway import ModelRateLimited

    app = await _client_with_script(
        settings,
        retriever,
        no_sleep,
        {
            "primary": [ModelRateLimited("primary: quota", retry_after=30.0)],
            "secondary": [ModelRateLimited("secondary: quota", retry_after=45.0)],
        },
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/reports", json={"question": "Digital trust in SEA"})

    assert r.status_code == 503
    assert r.headers["retry-after"] == "30", "should advertise the soonest reset"
    assert r.json()["error"] == "model_gateway_unavailable"


async def test_partial_failure_still_succeeds_via_fallback(settings, retriever, no_sleep):
    """One dead model must not fail the request when a sibling can serve it."""
    from digital_economy_agent.gateway import ModelUnavailable

    app = await _client_with_script(
        settings,
        retriever,
        no_sleep,
        {"primary": [ModelUnavailable("primary: HTTP 404 withdrawn")]},
    )
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/reports", json={"question": "Digital trust in SEA"})
            metrics = (await c.get("/metrics")).text

    assert r.status_code == 202
    assert r.json()["cost"]["models_used"] == ["groq/secondary"]
    assert _sample(metrics, "agent_gateway_model_retirements_total", model="groq/primary") >= 1.0
    assert _sample(metrics, "agent_gateway_fallbacks_total", model="groq/secondary") >= 1.0
