"""
Prometheus metrics.

Phase 1 exposed a JSON blob at ``/metrics``. Convenient to read, useless to scrape: no
Prometheus, Cloud Monitoring or Grafana can consume it, so nothing could alert on cost
or latency. This replaces it with the exposition format.

Metric design notes:

* **Labels are bounded.** Model names and node names are enumerable; query text,
  thread ids and user input never appear as labels. An unbounded label creates one time
  series per distinct value, which is how a metrics backend gets taken down.
* **Counters, not gauges, for cumulative work.** A counter survives a scrape gap and
  restarts visibly; a gauge of "total cost" silently resets to zero on redeploy.
* **Cost is metered per model**, because the JD's operational question is not "what did
  this cost" but "which model is costing us this".
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry as _Registry

# A dedicated registry rather than the global default. The default registry is process
# global, so two app instances in one test session would double-count and the tests
# would interfere with each other.
REGISTRY: _Registry = CollectorRegistry()

BUILD_INFO = Gauge(
    "agent_build_info",
    "Build metadata; the value is always 1 and the version lives in the label.",
    ["version"],
    registry=REGISTRY,
)

REQUESTS = Counter(
    "agent_requests_total",
    "HTTP requests handled, by route template and outcome class.",
    ["route", "status_class"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds",
    "Request latency by route template.",
    ["route"],
    # Buckets chosen for this workload: a report takes seconds because it makes two
    # sequential model calls, so the default buckets (ending at 10s) would put almost
    # every observation in +Inf and make p95 unreadable.
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0),
    registry=REGISTRY,
)

LLM_CALLS = Counter(
    "agent_llm_calls_total",
    "Model completions, by model and graph node.",
    ["model", "node"],
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "agent_llm_tokens_total",
    "Tokens consumed, by model and direction.",
    ["model", "direction"],
    registry=REGISTRY,
)

LLM_COST = Counter(
    "agent_llm_cost_usd_total",
    "Cumulative model spend in USD, by model.",
    ["model"],
    registry=REGISTRY,
)

GATEWAY_FALLBACKS = Counter(
    "agent_gateway_fallbacks_total",
    "Requests not served by the first model in the chain, by the model that served it.",
    ["model"],
    registry=REGISTRY,
)

GATEWAY_RETIREMENTS = Counter(
    "agent_gateway_model_retirements_total",
    "Models retired for the run after a permanent error, by model.",
    ["model"],
    registry=REGISTRY,
)

RETRIEVAL_CHUNKS = Histogram(
    "agent_retrieval_chunks",
    "Chunks returned per retrieval.",
    buckets=(0, 1, 2, 3, 4, 5, 8, 10, 20, 50),
    registry=REGISTRY,
)

RETRIEVAL_EMPTY = Counter(
    "agent_retrieval_empty_total",
    "Retrievals that returned nothing. A grounded system must report these rather "
    "than answer anyway, so a rising rate is a corpus or ingest problem.",
    registry=REGISTRY,
)

REVIEW_PAUSES = Counter(
    "agent_review_pauses_total",
    "Runs that reached the human review gate.",
    registry=REGISTRY,
)

REVIEW_DECISIONS = Counter(
    "agent_review_decisions_total",
    "Human review verdicts, by action.",
    ["action"],
    registry=REGISTRY,
)

REVISION_BUDGET_EXHAUSTED = Counter(
    "agent_revision_budget_exhausted_total",
    "Runs halted because the revision budget ran out.",
    registry=REGISTRY,
)


def status_class(status_code: int) -> str:
    """Bucket a status code as 2xx/4xx/5xx, keeping the label cardinality at three."""
    return f"{status_code // 100}xx"


def record_llm_call(
    *,
    model: str,
    node: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    fell_back: bool,
) -> None:
    """Record one model completion across the call, token and cost meters."""
    LLM_CALLS.labels(model=model, node=node).inc()
    LLM_TOKENS.labels(model=model, direction="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(model=model, direction="completion").inc(completion_tokens)
    LLM_COST.labels(model=model).inc(cost_usd)
    if fell_back:
        GATEWAY_FALLBACKS.labels(model=model).inc()


def render() -> bytes:
    """Prometheus exposition output for the ``/metrics`` endpoint."""
    return generate_latest(REGISTRY)


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
