"""
Observability tests.

Metrics and traces are the things nobody notices are broken until an incident, when it
is too late to find out. Counters are asserted as deltas because a Prometheus registry
is process-global and cumulative by design — an absolute assertion passes alone and
fails in a suite.
"""

from __future__ import annotations

from contextlib import suppress

import pytest

from digital_economy_agent.observability import configure_tracing, metrics, span


@pytest.fixture(autouse=True, scope="module")
def _quiet_span_exporter():
    """
    Flush and shut the tracer provider down at the end of this module.

    The ConsoleSpanExporter writes to the stdout pytest captures, and the batch
    processor flushes on a background thread -- after pytest has closed that stream.
    The result is an alarming "I/O operation on closed file" traceback in an otherwise
    passing run, which is exactly the kind of noise that trains people to ignore CI
    output.
    """
    yield
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    for method in ("force_flush", "shutdown"):
        hook = getattr(provider, method, None)
        if callable(hook):
            with suppress(Exception):
                hook()


def _value(body: str, metric: str, **labels: str) -> float:
    label_part = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    needle = f"{metric}{{{label_part}}}" if label_part else metric
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if name.strip() == needle:
            return float(value)
    return 0.0


# --- metrics ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [(200, "2xx"), (202, "2xx"), (404, "4xx"), (422, "4xx"), (500, "5xx"), (503, "5xx")],
)
def test_status_class_keeps_label_cardinality_at_three(code, expected):
    """One series per status *class*, not per code: bounded cardinality by design."""
    assert metrics.status_class(code) == expected


def test_record_llm_call_updates_every_meter():
    before = metrics.render().decode()
    metrics.record_llm_call(
        model="test/model",
        node="research",
        prompt_tokens=100,
        completion_tokens=40,
        cost_usd=0.000123,
        fell_back=False,
    )
    after = metrics.render().decode()

    def delta(metric: str, **labels: str) -> float:
        return _value(after, metric, **labels) - _value(before, metric, **labels)

    assert delta("agent_llm_calls_total", model="test/model", node="research") == 1.0
    assert delta("agent_llm_tokens_total", model="test/model", direction="prompt") == 100.0
    assert delta("agent_llm_tokens_total", model="test/model", direction="completion") == 40.0
    assert delta("agent_llm_cost_usd_total", model="test/model") == pytest.approx(0.000123)
    # No fallback happened, so that counter must stay untouched.
    assert delta("agent_gateway_fallbacks_total", model="test/model") == 0.0


def test_fallback_is_metered_separately_from_the_call():
    """A fallback is an operational event, not just another completion."""
    before = metrics.render().decode()
    metrics.record_llm_call(
        model="test/fallback",
        node="write_draft",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0,
        fell_back=True,
    )
    after = metrics.render().decode()
    assert (
        _value(after, "agent_gateway_fallbacks_total", model="test/fallback")
        - _value(before, "agent_gateway_fallbacks_total", model="test/fallback")
        == 1.0
    )


def test_render_is_prometheus_exposition_format():
    body = metrics.render().decode()
    assert "# HELP agent_llm_cost_usd_total" in body
    assert "# TYPE agent_llm_calls_total counter" in body
    assert metrics.CONTENT_TYPE.startswith("text/plain")


def test_latency_buckets_cover_a_multi_second_report():
    """
    A report makes two sequential model calls, so it takes seconds. The default
    histogram buckets stop at 10s, which would put nearly every observation in +Inf and
    make p95 unreadable.
    """
    upper = metrics.REQUEST_DURATION._upper_bounds
    assert max(b for b in upper if b != float("inf")) >= 60.0
    assert any(b <= 0.1 for b in upper), "fast ops (healthz, metrics) need low buckets too"


def test_metrics_use_a_dedicated_registry():
    """
    Not the process-global default registry.

    Two app instances in one test session would otherwise double-count, and the tests
    would interfere with each other rather than merely accumulate.
    """
    from prometheus_client import REGISTRY as DEFAULT_REGISTRY

    assert metrics.REGISTRY is not DEFAULT_REGISTRY
    assert "agent_llm_calls_total" not in DEFAULT_REGISTRY._names_to_collectors


# --- tracing ---------------------------------------------------------------


def test_tracing_is_disabled_without_an_endpoint():
    """
    Opt-in, not best-effort.

    An exporter pointing at a collector that is not there retries in the background and
    adds latency to every request, turning a missing telemetry sidecar into a
    user-visible problem.
    """
    assert configure_tracing(service_name="t", service_version="0", otlp_endpoint="") is False


def test_tracing_enables_with_the_console_exporter():
    assert configure_tracing(service_name="t", service_version="0", console=True) is True


def test_span_records_an_exception_as_an_error_status():
    """
    Without an explicit status an errored span renders as successful in a trace view,
    which makes the trace worse than no trace: it actively misleads.
    """
    configure_tracing(service_name="t", service_version="0", console=True)
    with pytest.raises(ValueError, match="boom"), span("failing-operation") as current:
        recorded = current
        raise ValueError("boom")

    from opentelemetry.trace import StatusCode

    assert recorded.status.status_code is StatusCode.ERROR


def test_span_attaches_attributes_and_skips_none():
    configure_tracing(service_name="t", service_version="0", console=True)
    with span("op", kept="value", dropped=None) as current:
        assert current.attributes is not None
        assert current.attributes.get("kept") == "value"
        assert "dropped" not in current.attributes
