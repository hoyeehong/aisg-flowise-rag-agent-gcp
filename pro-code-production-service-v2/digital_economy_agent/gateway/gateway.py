"""
Model gateway: routing, rate-limit handling, retries and fallback chains.

Selection is *sticky*. Once a model serves a request it keeps serving subsequent ones,
for two reasons: re-probing a chain from the top costs latency on every call, and — for
evaluation especially — a model that silently oscillates makes aggregate scores a blend
of different models rather than a measurement of one.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .errors import (
    AllModelsFailed,
    ModelRateLimited,
    ModelTransient,
    ModelUnavailable,
)
from .types import ChatMessage, ChatProvider, Completion, ModelSpec, TokenUsage

logger = logging.getLogger(__name__)

# Injectable so tests exercise backoff arithmetic without real delays.
SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    """
    Bounded retry behaviour.

    ``max_attempts_per_model`` applies to transient faults only. Rate limiting is never
    retried against the same model: quota does not return in seconds, so the next model
    in the chain is a better use of the time.
    """

    max_attempts_per_model: int = 3
    backoff_base_seconds: float = 1.0
    max_wait_seconds: float = 30.0
    # When the whole chain is rate-limited there is nothing to fall back to, so wait
    # once — but only if the provider says the reset is within this budget. A longer
    # reset implies a daily quota, which should fail fast instead of hanging a request.
    cooldown_budget_seconds: float = 60.0

    def backoff_for(self, attempt: int, suggested: float | None) -> float:
        if suggested is not None:
            return float(min(suggested, self.max_wait_seconds))
        wait = float(min(self.backoff_base_seconds * (2 ** (attempt - 1)), self.max_wait_seconds))
        return wait + random.uniform(0, 0.5 * wait)  # jitter: avoid lockstep retries


@dataclass
class GatewayStats:
    """Per-gateway accounting, exposed for the /metrics surface and cost meters."""

    requests: int = 0
    fallbacks: int = 0
    total_cost_usd: float = 0.0
    calls_by_model: dict[str, int] = field(default_factory=dict)
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    retired: dict[str, str] = field(default_factory=dict)

    def record(self, spec: ModelSpec, usage: TokenUsage, cost: float, *, fell_back: bool) -> None:
        self.requests += 1
        if fell_back:
            self.fallbacks += 1
        self.calls_by_model[spec.key] = self.calls_by_model.get(spec.key, 0) + 1
        self.tokens_by_model[spec.key] = self.tokens_by_model.get(spec.key, 0) + usage.total_tokens
        self.cost_by_model[spec.key] = self.cost_by_model.get(spec.key, 0.0) + cost
        self.total_cost_usd += cost


class ModelGateway:
    """Routes chat completions across an ordered chain of models."""

    def __init__(
        self,
        chain: list[ModelSpec],
        providers: dict[str, ChatProvider],
        *,
        policy: RetryPolicy | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        if not chain:
            raise ValueError("model chain must not be empty")
        missing = sorted({s.provider for s in chain} - set(providers))
        if missing:
            raise ValueError(f"no provider registered for: {', '.join(missing)}")
        self.chain = list(chain)
        self.providers = providers
        self.policy = policy or RetryPolicy()
        self.stats = GatewayStats()
        self._rate_limited: dict[str, float | None] = {}
        self._active: ModelSpec | None = None
        self._cooldowns_used = 0
        self._sleep: SleepFn = sleep or asyncio.sleep

    @property
    def active_model(self) -> str | None:
        return self._active.key if self._active else None

    def _candidates(self) -> list[ModelSpec]:
        ordered = ([self._active] if self._active else []) + [
            s for s in self.chain if s is not self._active
        ]
        return [s for s in ordered if s.key not in self.stats.retired]

    async def _try_model(
        self,
        spec: ModelSpec,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage, int]:
        """Attempt one model, retrying transient faults only. Returns (text, usage, attempts)."""
        provider = self.providers[spec.provider]
        last: ModelTransient | None = None
        for attempt in range(1, self.policy.max_attempts_per_model + 1):
            try:
                text, usage = await provider.complete(
                    spec.model, messages, temperature=temperature, max_tokens=max_tokens
                )
                return text, usage, attempt
            except ModelRateLimited as exc:
                self._rate_limited[spec.key] = exc.retry_after
                raise
            except ModelUnavailable:
                raise
            except ModelTransient as exc:
                last = exc
                if attempt == self.policy.max_attempts_per_model:
                    break
                if exc.retry_after is not None and exc.retry_after > self.policy.max_wait_seconds:
                    break
                wait = self.policy.backoff_for(attempt, exc.retry_after)
                logger.warning(
                    "model %s transient failure (attempt %d/%d), retrying in %.1fs: %s",
                    spec.key,
                    attempt,
                    self.policy.max_attempts_per_model,
                    wait,
                    exc,
                )
                await self._sleep(wait)
        assert last is not None
        raise last

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Completion:
        """Complete a chat request, walking the chain until a model answers."""
        started = time.perf_counter()
        try:
            return await self._walk_chain(messages, temperature, max_tokens, started)
        except AllModelsFailed as first:
            waits = [d for d in self._rate_limited.values() if d is not None]
            if not waits or self._cooldowns_used >= 1:
                raise
            wait = min(waits)
            if wait > self.policy.cooldown_budget_seconds:
                raise AllModelsFailed(
                    f"{first} — soonest quota reset is ~{wait:.0f}s, beyond the "
                    f"{self.policy.cooldown_budget_seconds:.0f}s cooldown budget; "
                    f"this looks like a daily quota rather than a burst limit",
                    first.attempts,
                    categories=first.categories,
                    retry_after=wait,
                ) from first
            self._cooldowns_used += 1
            logger.warning("entire model chain rate-limited; cooling down %.0fs", wait)
            await self._sleep(wait + 1.0)
            self._rate_limited.clear()
            return await self._walk_chain(messages, temperature, max_tokens, started)

    async def _walk_chain(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        started: float,
    ) -> Completion:
        candidates = self._candidates()
        if not candidates:
            raise AllModelsFailed(
                "every model in the chain has been retired",
                dict(self.stats.retired),
                categories={k: "unavailable" for k in self.stats.retired},
            )

        first_choice = candidates[0]
        attempts: dict[str, str] = {}
        categories: dict[str, str] = {}
        for spec in candidates:
            try:
                text, usage, tries = await self._try_model(spec, messages, temperature, max_tokens)
            except ModelUnavailable as exc:
                self.stats.retired[spec.key] = str(exc)
                attempts[spec.key] = f"unavailable: {exc}"
                categories[spec.key] = "unavailable"
                logger.error("retiring model %s for this run: %s", spec.key, exc)
                continue
            except ModelRateLimited as exc:
                attempts[spec.key] = f"rate limited: {exc}"
                categories[spec.key] = "rate_limited"
                logger.warning("model %s rate limited, advancing chain", spec.key)
                continue
            except ModelTransient as exc:
                attempts[spec.key] = f"transient: {exc}"
                categories[spec.key] = "transient"
                logger.warning("model %s still failing, advancing chain", spec.key)
                continue

            fell_back = spec is not first_choice
            if self._active is not None and spec is not self._active:
                logger.warning("gateway switched model: %s -> %s", self._active.key, spec.key)
            self._active = spec
            cost = spec.cost_usd(usage)
            self.stats.record(spec, usage, cost, fell_back=fell_back)
            return Completion(
                text=text,
                model=spec.key,
                usage=usage,
                cost_usd=cost,
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=tries,
                fell_back=fell_back,
            )

        waits = [d for d in self._rate_limited.values() if d is not None]
        raise AllModelsFailed(
            "all models in the chain failed: "
            + "; ".join(f"{k} ({v})" for k, v in attempts.items()),
            attempts,
            categories=categories,
            retry_after=min(waits) if waits else None,
        )
