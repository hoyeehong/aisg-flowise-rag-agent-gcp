"""
OpenTelemetry tracing.

Disabled unless an OTLP endpoint is configured. That default matters: an exporter
pointing at a collector that is not there retries in the background and turns a missing
dependency into latency on every request, so tracing must be opt-in rather than
best-effort.

Spans are added where the questions actually get asked in production -- which model
served a request, how long retrieval took, how many revisions a run needed -- rather
than on every function.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

_TRACER_NAME = "digital_economy_agent"


def configure_tracing(
    *,
    service_name: str,
    service_version: str,
    otlp_endpoint: str = "",
    console: bool = False,
    environment: str = "development",
) -> bool:
    """
    Install a tracer provider. Returns whether tracing was actually enabled.

    Failure to construct the exporter is logged and swallowed: a missing collector must
    degrade observability, never the service. Losing traces is an inconvenience;
    refusing requests because a telemetry sidecar is down is an outage.
    """
    if not otlp_endpoint and not console:
        logger.info("tracing disabled: no OTLP endpoint configured")
        return False

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
            )
            logger.info("tracing enabled, exporting to %s", otlp_endpoint)
        except Exception as exc:
            logger.error("OTLP exporter unavailable, tracing degraded: %s", exc)
    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return True


def tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """
    Start a span, recording an exception as the span status if one escapes.

    Without the explicit status an errored span looks successful in a trace view, which
    makes the trace worse than no trace: it actively misleads.
    """
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except Exception as exc:
            current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            current.record_exception(exc)
            raise


def instrument_app(app: Any) -> None:
    """
    Auto-instrument FastAPI and httpx.

    Health and metrics endpoints are excluded: a scrape every 15s and a probe every 10s
    would otherwise dominate the trace volume and the bill, while telling nobody
    anything.
    """
    with suppress(Exception):
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    with suppress(Exception):
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
