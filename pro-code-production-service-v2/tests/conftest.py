"""Shared fixtures: fake providers and a small corpus, so tests need no network."""

from __future__ import annotations

import pytest

from digital_economy_agent.gateway import (
    ChatMessage,
    ModelGateway,
    ModelSpec,
    RetryPolicy,
    TokenUsage,
)
from digital_economy_agent.tools import Chunk, InMemoryRetriever


class FakeProvider:
    """
    Scripted provider.

    ``script`` maps a model name to a list of outcomes consumed in order. An outcome is
    either a string (returned as the completion) or an exception instance (raised), which
    lets a test drive the gateway's retry and fallback paths deterministically.
    """

    def __init__(self, name: str, script: dict[str, list[object]] | None = None) -> None:
        self.name = name
        self.script = script or {}
        self.calls: list[tuple[str, int]] = []

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        self.calls.append((model, len(messages)))
        outcomes = self.script.get(model)
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            text = str(outcome)
        else:
            text = f"[{model}] response to: {messages[-1].content[:60]}"
        return text, TokenUsage(prompt_tokens=100, completion_tokens=50)


@pytest.fixture
def no_sleep():
    """Replace backoff sleeps so retry tests run instantly, recording requested waits."""
    waits: list[float] = []

    async def _sleep(seconds: float) -> None:
        waits.append(seconds)

    _sleep.waits = waits  # type: ignore[attr-defined]
    return _sleep


@pytest.fixture
def specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            provider="fake",
            model="primary",
            usd_per_million_input=0.10,
            usd_per_million_output=0.40,
        ),
        ModelSpec(
            provider="fake",
            model="secondary",
            usd_per_million_input=0.05,
            usd_per_million_output=0.20,
        ),
    ]


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider("fake")


@pytest.fixture
def gateway(specs, provider, no_sleep) -> ModelGateway:
    return ModelGateway(
        specs,
        {"fake": provider},
        policy=RetryPolicy(max_attempts_per_model=3, backoff_base_seconds=1.0),
        sleep=no_sleep,
    )


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        Chunk(
            text=(
                "Southeast Asia's digital economy is shifting from Tech for Growth to "
                "Tech for Good, prioritising inclusion and trust over raw GMV expansion."
            ),
            source="imda_report.pdf",
            page=4,
        ),
        Chunk(
            text=(
                "Digital trust and governance require proactive cybersecurity, privacy "
                "protection and responsible AI frameworks across the SEA-6 economies."
            ),
            source="imda_report.pdf",
            page=18,
        ),
        Chunk(
            text=(
                "Digital talent shortages constrain Indonesia and Vietnam, where demand "
                "for advanced skills outpaces training pipeline capacity."
            ),
            source="imda_report.pdf",
            page=23,
        ),
    ]


@pytest.fixture
def retriever(corpus) -> InMemoryRetriever:
    return InMemoryRetriever(corpus)
