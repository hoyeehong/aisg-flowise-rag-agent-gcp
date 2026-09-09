"""
HTTP client for the service under test.

The harness drives the deployed API rather than importing the graph. That is the whole
point of Phase 3: v1's suite scored a pre-recorded fixture, so a regression in the
running system produced identical scores. Anything reachable over HTTP -- a local
uvicorn, a container, a Cloud Run URL -- can be evaluated by the same harness.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# 503 from this service means its model gateway is rate limited, not that it is broken.
# Retrying is essential: without it the harness measures the provider's tokens-per-minute
# ceiling rather than the system. Groq's free tier is 8k TPM against ~2.5k tokens per
# report, so an unpaced run exhausts the budget after three cases.
RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class ServiceError(RuntimeError):
    """The service under test did not return a usable response."""


class ServiceRateLimited(ServiceError):
    """The service is rate limited and did not recover within the retry budget."""


@dataclass(frozen=True)
class RunOutcome:
    """One report run, as observed over HTTP."""

    thread_id: str
    answer: str
    citations: list[str]
    context: str
    retrieved_chars: int
    latency_ms: float
    cost_usd: float
    status: str


class ServiceClient:
    """Minimal client for the report endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 180.0,
        max_attempts: int = 4,
        backoff_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds

    async def readyz(self) -> tuple[int, dict[str, object]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/readyz")
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {}

    async def corpus_coverage(self) -> dict[str, list[int]]:
        """Indexed pages per source, used as a retrieval-scoring precondition."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/v1/corpus")
        if response.status_code != 200:
            raise ServiceError(f"/v1/corpus HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        return {str(k): [int(p) for p in v] for k, v in (body.get("sources") or {}).items()}

    async def retrieve_only(self, query: str, *, top_k: int = 5) -> RunOutcome:
        """
        Score retrieval without generating a report.

        No chat model is invoked, so with the deterministic embedder this path needs no
        API key and no quota -- which is what makes retrieval metrics gateable on every
        pull request.
        """
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/retrieve", json={"query": query, "top_k": top_k}
            )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code >= 400:
            raise ServiceError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        return RunOutcome(
            thread_id="",
            answer="",
            citations=list(body.get("citations") or []),
            context=body.get("context") or "",
            retrieved_chars=int(body.get("retrieved_chars") or 0),
            latency_ms=latency_ms,
            cost_usd=0.0,
            status="retrieval-only",
        )

    async def run_case(self, query: str, *, top_k: int = 5) -> RunOutcome:
        """
        Start a report and return the draft awaiting review.

        Evaluation stops at the human review gate deliberately: the draft is the model's
        unassisted output. Approving first and scoring the final report would measure a
        human-corrected answer and flatter the system.
        """
        payload = {
            # include_context is what makes groundedness scoreable: the judge must see
            # the same context the model saw, not an empty string.
            "question": query,
            "top_k": top_k,
            "include_context": True,
        }

        response: httpx.Response | None = None
        latency_ms = 0.0
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.base_url}/v1/reports", json=payload)
            latency_ms = (time.perf_counter() - started) * 1000

            if response.status_code not in RETRYABLE_STATUS:
                break
            if attempt == self._max_attempts:
                raise ServiceRateLimited(
                    f"HTTP {response.status_code} after {attempt} attempt(s): {response.text[:200]}"
                )
            # Prefer the service's own advice; it forwards the provider's reset window.
            advised = response.headers.get("retry-after")
            try:
                wait = float(advised) if advised else self._backoff * attempt
            except ValueError:
                wait = self._backoff * attempt
            wait += random.uniform(0, 0.25 * wait)
            logger.warning(
                "service rate limited (HTTP %s), retry %d/%d in %.0fs",
                response.status_code,
                attempt,
                self._max_attempts,
                wait,
            )
            await asyncio.sleep(wait)

        assert response is not None
        if response.status_code >= 400:
            raise ServiceError(f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise ServiceError(f"non-JSON response: {response.text[:200]}") from exc

        answer = body.get("draft") or body.get("final_report") or ""
        if not answer:
            raise ServiceError(f"response carried no draft: {str(body)[:200]}")

        return RunOutcome(
            thread_id=body.get("thread_id", ""),
            answer=answer,
            citations=list(body.get("citations") or []),
            context=body.get("context") or "",
            retrieved_chars=int(body.get("retrieved_chars") or 0),
            latency_ms=latency_ms,
            cost_usd=float((body.get("cost") or {}).get("total_cost_usd") or 0.0),
            status=body.get("status", ""),
        )
