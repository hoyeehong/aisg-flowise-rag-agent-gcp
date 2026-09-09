"""FastAPI application: routes, dependency wiring and error translation."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any, cast

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .. import __version__
from ..agents import build_graph
from ..config import Settings, get_settings
from ..gateway import AllModelsFailed, GatewayError, ModelGateway, OpenAICompatibleProvider
from ..retrieval import (
    Embedder,
    GoogleEmbedder,
    HashingEmbedder,
    HybridRetriever,
    PgVectorStore,
)
from ..tools import Chunk, InMemoryRetriever, Retriever
from .schemas import (
    CreateReportRequest,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    ReportState,
    ReviewRequest,
)
from .service import ReportService, RunNotAwaitingReviewError, RunNotFoundError

logger = logging.getLogger(__name__)

# Development corpus for the in-memory retriever. Phase 2 replaces this with the
# ingestion pipeline's pgvector index; the Retriever protocol does not change.
_DEV_CORPUS = [
    Chunk(
        text=(
            "Southeast Asia's digital economy is shifting from Tech for Growth to Tech for "
            "Good, prioritising inclusion, trust and sustainability alongside growth in "
            "gross merchandise value."
        ),
        source="imda_report.pdf",
        page=4,
    ),
    Chunk(
        text=(
            "Digital trust and governance depend on proactive cybersecurity, consumer "
            "safety, privacy protection and responsible AI frameworks. Singapore's AI "
            "governance stance is cited as a regional reference point."
        ),
        source="imda_report.pdf",
        page=18,
    ),
    Chunk(
        text=(
            "Digital talent supply constrains Indonesia, Vietnam and the Philippines, where "
            "demand for advanced digital skills outpaces the training pipeline."
        ),
        source="imda_report.pdf",
        page=23,
    ),
    Chunk(
        text=(
            "ASEAN's Digital Economy Framework Agreement (DEFA) aims to harmonise "
            "cross-border data flows, digital payments and regulatory cohesion across "
            "member states."
        ),
        source="imda_report.pdf",
        page=31,
    ),
]


def build_retriever(settings: Settings, *, gateway: ModelGateway | None = None) -> Retriever:
    """
    Select a retrieval implementation.

    Both satisfy the same Retriever protocol, so nothing in agents/ or api/ changes
    between them -- which is what the protocol was declared for in Phase 1.
    """
    if settings.use_in_memory_retriever:
        return InMemoryRetriever(_DEV_CORPUS)

    if not settings.postgres_configured:
        raise ValueError(
            "pgvector retrieval requires AGENT_POSTGRES_DSN "
            "(or set AGENT_USE_IN_MEMORY_RETRIEVER=true)"
        )
    embedder: Embedder
    if settings.has_embedding_credentials:
        embedder = GoogleEmbedder(
            settings.gemini_api_key.get_secret_value(),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    else:
        # Explicit and loud: the stack works, but retrieval quality will be lexical.
        logger.warning(
            "no embedding credentials; falling back to the hashing embedder. Retrieval "
            "will have no semantic recall. Set GEMINI_API_KEY for real embeddings."
        )
        embedder = HashingEmbedder(dimensions=settings.embedding_dimensions)

    store = PgVectorStore(settings.postgres_dsn, dimensions=settings.embedding_dimensions)
    return HybridRetriever(
        store,
        embedder,
        config=settings.hybrid_config,
        gateway=gateway,
        tenant_id=settings.tenant_id,
    )


def build_gateway(settings: Settings, *, client: httpx.AsyncClient | None = None) -> ModelGateway:
    """Construct the model gateway from settings."""
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key.get_secret_value(),
        client=client,
        timeout=settings.request_timeout_seconds,
    )
    return ModelGateway(settings.models, {"groq": provider})


def get_service(request: Request) -> ReportService:
    """Resolve the per-app service from application state."""
    return cast(ReportService, request.app.state.service)


# Must live at module scope: annotations are strings under `from __future__ import
# annotations`, and FastAPI resolves them in this module's namespace. An alias defined
# inside create_app() would not resolve, and `service` would be read as a query
# parameter -- which surfaces as a confusing 422 on every request.
ServiceDep = Annotated[ReportService, Depends(get_service)]


def create_app(
    settings: Settings | None = None,
    *,
    gateway: ModelGateway | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """
    Build the application.

    ``gateway`` and ``retriever`` are injectable so tests exercise the real routes and
    the real graph against scripted providers, rather than mocking the routes away.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=resolved.log_level.upper())
        app.state.settings = resolved
        app.state.gateway = gateway or build_gateway(resolved)
        app.state.retriever = retriever or build_retriever(resolved, gateway=app.state.gateway)

        # Durable human-review pauses need a real checkpointer. An injected retriever
        # means a test harness, so the in-process saver is correct there.
        checkpointer: BaseCheckpointSaver[Any] = InMemorySaver()
        app.state.checkpoint_backend = "memory"
        app.state.exit_stack = AsyncExitStack()
        if (
            retriever is None
            and resolved.use_postgres_checkpointer
            and resolved.postgres_configured
        ):
            saver = await app.state.exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(resolved.postgres_dsn)
            )
            await saver.setup()
            checkpointer = saver
            app.state.checkpoint_backend = "postgres"
        elif retriever is None and resolved.use_postgres_checkpointer:
            logger.warning(
                "AGENT_POSTGRES_DSN is unset; paused human-review runs will be lost on "
                "restart. Set it to make the review gate durable."
            )

        if isinstance(app.state.retriever, HybridRetriever):
            await PgVectorStore(
                resolved.postgres_dsn, dimensions=resolved.embedding_dimensions
            ).ensure_schema()

        app.state.graph = build_graph(
            app.state.gateway, app.state.retriever, checkpointer=checkpointer
        )
        app.state.service = ReportService(
            app.state.graph,
            default_top_k=resolved.default_top_k,
            max_revisions=resolved.max_revisions,
        )
        describe = getattr(app.state.retriever, "describe", None)
        logger.info(
            "service ready: models=%s retriever=%s checkpointer=%s",
            [s.key for s in resolved.models],
            describe() if callable(describe) else type(app.state.retriever).__name__,
            app.state.checkpoint_backend,
        )
        try:
            yield
        finally:
            await app.state.exit_stack.aclose()

    app = FastAPI(
        title="Digital Economy Research & Report Agent",
        version=__version__,
        summary="Multi-agent RAG service with a human-in-the-loop review gate.",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness: the process is up. Deliberately checks no dependency."""
        return HealthResponse(version=__version__)

    @app.get(
        "/readyz",
        response_model=ReadyResponse,
        tags=["ops"],
        responses={503: {"model": ReadyResponse, "description": "Not ready to serve traffic"}},
    )
    async def readyz(request: Request, response: Response) -> ReadyResponse:
        """
        Readiness: whether this instance can actually serve a request.

        Returns **503 when degraded**, not 200. A readiness probe decides routing from
        the status code alone -- Kubernetes and Cloud Run do not parse the body -- so
        answering 200 while reporting `"status": "degraded"` keeps traffic arriving at
        an instance that has just said it cannot do its job. With
        `durable_checkpointer: false`, for example, a restart would silently discard
        work awaiting a human reviewer.

        Deliberately distinct from `/healthz`, which stays 200 whenever the process is
        up: a failing *liveness* probe triggers a restart, and restarting cannot conjure
        a missing credential or database. Readiness drains; liveness restarts.
        """
        cfg: Settings = request.app.state.settings
        checks = {
            "graph_compiled": request.app.state.graph is not None,
            "retriever_ready": request.app.state.retriever is not None,
            "model_credentials": cfg.has_model_credentials,
        }
        if not cfg.use_in_memory_retriever:
            # Only assert these when the service is actually configured to need them.
            checks["embedding_credentials"] = cfg.has_embedding_credentials
            checks["postgres_configured"] = cfg.postgres_configured
        if cfg.use_postgres_checkpointer:
            checks["durable_checkpointer"] = request.app.state.checkpoint_backend == "postgres"
        ok = all(checks.values())
        if not ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="ready" if ok else "degraded",
            checks=checks,
            detail="" if ok else "missing: " + ", ".join(k for k, v in checks.items() if not v),
        )

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: Request) -> dict[str, Any]:
        """
        Gateway counters, including a cost meter per model.

        JSON for now; Phase 4 exposes these in Prometheus exposition format alongside
        OpenTelemetry traces.
        """
        stats = request.app.state.gateway.stats
        return {
            "requests_total": stats.requests,
            "fallbacks_total": stats.fallbacks,
            "cost_usd_total": round(stats.total_cost_usd, 8),
            "calls_by_model": stats.calls_by_model,
            "tokens_by_model": stats.tokens_by_model,
            "cost_usd_by_model": {k: round(v, 8) for k, v in stats.cost_by_model.items()},
            "retired_models": sorted(stats.retired),
            "active_model": request.app.state.gateway.active_model,
        }

    @app.post(
        "/v1/reports",
        response_model=ReportState,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["reports"],
    )
    async def create_report(body: CreateReportRequest, service: ServiceDep) -> ReportState:
        """
        Start a run. Returns 202 because the run pauses at the human review gate
        rather than completing: the response carries a draft awaiting a verdict.
        """
        thread_id = str(uuid.uuid4())
        return await service.start(
            thread_id, body.question, top_k=body.top_k, max_revisions=body.max_revisions
        )

    @app.get("/v1/reports/{thread_id}", response_model=ReportState, tags=["reports"])
    async def get_report(thread_id: str, service: ServiceDep) -> ReportState:
        try:
            return await service.get(thread_id)
        except RunNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {thread_id}") from exc

    @app.post("/v1/reports/{thread_id}/review", response_model=ReportState, tags=["reports"])
    async def review_report(
        thread_id: str, body: ReviewRequest, service: ServiceDep
    ) -> ReportState:
        """Approve the draft, or request a revision with feedback."""
        try:
            return await service.review(thread_id, body.action, body.feedback)
        except RunNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run {thread_id}") from exc
        except RunNotAwaitingReviewError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except ValueError as exc:
            # e.g. a revision request with no feedback.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    @app.exception_handler(AllModelsFailed)
    async def all_models_failed(request: Request, exc: Exception) -> JSONResponse:
        """
        Translate a total gateway failure into an accurate status code.

        This is a *dependency* failure, not a bug in this service, so 500 is wrong: it
        tells a caller to report a defect and tells an SLO dashboard this service is
        broken. A capacity problem the client should retry is 503 with Retry-After;
        anything else (bad credential, withdrawn model) is 502.
        """
        failure = exc if isinstance(exc, AllModelsFailed) else None
        retryable = bool(failure and failure.rate_limited_only)
        logger.error("model gateway exhausted: %s", exc)

        headers: dict[str, str] = {}
        if retryable and failure and failure.retry_after:
            headers["Retry-After"] = str(int(failure.retry_after))

        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE if retryable else status.HTTP_502_BAD_GATEWAY
            ),
            content=ErrorResponse(
                error="model_gateway_unavailable",
                detail=(
                    "Every model in the fallback chain is rate limited; retry shortly."
                    if retryable
                    else "No model in the fallback chain could serve this request."
                ),
            ).model_dump(),
            headers=headers,
        )

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all for gateway faults that escape a node without being AllModelsFailed."""
        logger.error("model gateway error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error="model_gateway_error", detail="The model provider could not be reached."
            ).model_dump(),
        )

    @app.post("/v1/reports/stream", tags=["reports"])
    async def create_report_streaming(
        body: CreateReportRequest, request: Request
    ) -> StreamingResponse:
        """
        Start a run, streaming node completions as server-sent events.

        Useful because the research and drafting steps take tens of seconds: the client
        sees progress instead of one long silence. The terminal event carries the same
        payload as POST /v1/reports.
        """
        thread_id = str(uuid.uuid4())
        graph = request.app.state.graph
        service: ReportService = request.app.state.service
        settings: Settings = request.app.state.settings

        async def events() -> AsyncIterator[str]:
            def sse(event: str, data: dict[str, Any]) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            yield sse("started", {"thread_id": thread_id})
            try:
                async for update in graph.astream(
                    {
                        "question": body.question,
                        "top_k": body.top_k or settings.default_top_k,
                        "max_revisions": (
                            settings.max_revisions
                            if body.max_revisions is None
                            else body.max_revisions
                        ),
                    },
                    config={"configurable": {"thread_id": thread_id}},
                    stream_mode="updates",
                ):
                    for node in update:
                        yield sse("node", {"node": node})
                yield sse("state", (await service.get(thread_id)).model_dump())
            except Exception as exc:  # surface failures in-band; the response is already 200
                logger.exception("streaming run %s failed", thread_id)
                yield sse("error", {"error": type(exc).__name__, "detail": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# Served with `uvicorn --factory digital_economy_agent.api.app:create_app`. Using the
# factory rather than a module-level instance means importing this module in tests --
# which build their own app with scripted providers -- never needs real credentials.
__all__ = ["build_gateway", "build_retriever", "create_app"]
