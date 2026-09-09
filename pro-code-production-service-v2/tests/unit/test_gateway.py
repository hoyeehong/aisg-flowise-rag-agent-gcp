"""
Gateway tests.

The point of these is the *taxonomy*: rate limiting, transient faults and permanent
errors must produce three different behaviours. Treating them alike is what made the
v1 judge sleep 42s on a quota error before giving up.
"""

from __future__ import annotations

import pytest

from digital_economy_agent.gateway import (
    AllModelsFailed,
    ChatMessage,
    ModelGateway,
    ModelRateLimited,
    ModelSpec,
    ModelTransient,
    ModelUnavailable,
    RetryPolicy,
)

from ..conftest import FakeProvider

MSG = [ChatMessage(role="user", content="hello")]


def _gateway(script, specs, no_sleep, **policy_kw):
    provider = FakeProvider("fake", script)
    gw = ModelGateway(
        specs,
        {"fake": provider},
        policy=RetryPolicy(**policy_kw) if policy_kw else RetryPolicy(),
        sleep=no_sleep,
    )
    return gw, provider


async def test_rate_limit_advances_immediately_without_sleeping(specs, no_sleep):
    """
    A 429 must not sleep. Quota returns in minutes, so waiting stalls the caller while
    a sibling model is very likely free.
    """
    gw, provider = _gateway(
        {"primary": [ModelRateLimited("primary: quota", retry_after=45.0)]}, specs, no_sleep
    )
    result = await gw.complete(MSG)

    assert result.model == "fake/secondary"
    assert result.fell_back is True
    assert no_sleep.waits == [], "a rate limit must not trigger backoff"
    # Exactly one attempt on the limited model, then straight to the next.
    assert [m for m, _ in provider.calls] == ["primary", "secondary"]


async def test_transient_fault_retries_same_model_with_backoff(specs, no_sleep):
    """A 5xx genuinely clears in seconds, so the same model deserves another attempt."""
    gw, _ = _gateway(
        {"primary": [ModelTransient("primary: 503"), ModelTransient("primary: 503"), "recovered"]},
        specs,
        no_sleep,
    )
    result = await gw.complete(MSG)

    assert result.text == "recovered"
    assert result.model == "fake/primary"
    assert result.fell_back is False
    assert result.attempts == 3
    assert len(no_sleep.waits) == 2, "should back off between attempts"
    assert no_sleep.waits[1] > no_sleep.waits[0], "backoff must grow"


async def test_permanent_error_retires_model_for_the_run(specs, no_sleep):
    """A 404/403 never succeeds, so it must not be re-probed on every later request."""
    gw, provider = _gateway(
        {"primary": [ModelUnavailable("primary: HTTP 404 unknown model")]}, specs, no_sleep
    )
    first = await gw.complete(MSG)
    second = await gw.complete(MSG)

    assert first.model == second.model == "fake/secondary"
    assert "fake/primary" in gw.stats.retired
    # 'primary' was tried once, ever -- not once per request.
    assert [m for m, _ in provider.calls].count("primary") == 1


async def test_selection_is_sticky_after_fallback(specs, no_sleep):
    """
    Once a model answers it keeps serving. Re-probing the chain per request costs
    latency, and for evaluation it lets judge identity oscillate mid-run.
    """
    gw, provider = _gateway(
        {"primary": [ModelRateLimited("primary: quota", retry_after=30.0)]}, specs, no_sleep
    )
    await gw.complete(MSG)
    provider.calls.clear()
    await gw.complete(MSG)

    assert gw.active_model == "fake/secondary"
    assert [m for m, _ in provider.calls] == ["secondary"], "must not re-probe primary"


async def test_all_models_failed_raises_with_per_model_detail(specs, no_sleep):
    gw, _ = _gateway(
        {
            "primary": [ModelTransient("primary: 503")] * 3,
            "secondary": [ModelTransient("secondary: 503")] * 3,
        },
        specs,
        no_sleep,
        max_attempts_per_model=3,
    )
    with pytest.raises(AllModelsFailed) as exc:
        await gw.complete(MSG)

    assert set(exc.value.attempts) == {"fake/primary", "fake/secondary"}


async def test_whole_chain_rate_limited_cools_down_once_then_succeeds(specs, no_sleep):
    """With nothing left to fall back to, one bounded wait beats failing the request."""
    gw, _ = _gateway(
        {
            "primary": [ModelRateLimited("primary: quota", retry_after=20.0), "after cooldown"],
            "secondary": [ModelRateLimited("secondary: quota", retry_after=25.0)],
        },
        specs,
        no_sleep,
        cooldown_budget_seconds=60.0,
    )
    result = await gw.complete(MSG)

    assert result.text == "after cooldown"
    # Waited the soonest reset (20s) plus a small buffer.
    assert no_sleep.waits == [21.0]


async def test_cooldown_refused_when_reset_exceeds_budget(specs, no_sleep):
    """A long reset means a daily quota; fail fast rather than hanging the request."""
    gw, _ = _gateway(
        {
            "primary": [ModelRateLimited("primary: quota", retry_after=3600.0)],
            "secondary": [ModelRateLimited("secondary: quota", retry_after=3600.0)],
        },
        specs,
        no_sleep,
        cooldown_budget_seconds=60.0,
    )
    with pytest.raises(AllModelsFailed, match="daily quota"):
        await gw.complete(MSG)

    assert no_sleep.waits == [], "must not sleep on a daily quota"


async def test_cost_is_metered_per_model(specs, no_sleep):
    gw, _ = _gateway({}, specs, no_sleep)
    result = await gw.complete(MSG)

    expected = (100 * 0.10 + 50 * 0.40) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)
    assert gw.stats.total_cost_usd == pytest.approx(expected)
    assert gw.stats.calls_by_model == {"fake/primary": 1}
    assert gw.stats.tokens_by_model == {"fake/primary": 150}


def test_empty_chain_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        ModelGateway([], {})


def test_unregistered_provider_is_rejected():
    spec = ModelSpec(provider="nope", model="m")
    with pytest.raises(ValueError, match="no provider registered"):
        ModelGateway([spec], {})
