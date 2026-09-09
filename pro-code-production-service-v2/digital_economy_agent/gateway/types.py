"""Typed contracts for the model gateway."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One turn in a chat completion request."""

    model_config = {"frozen": True}

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token counts for a single completion."""

    model_config = {"frozen": True}

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelSpec(BaseModel):
    """
    A callable model plus the pricing needed to meter it.

    Prices are USD per million tokens. They are declared here rather than fetched so a
    cost figure is always reproducible from the code that produced it; update them
    deliberately when a provider changes rates.
    """

    model_config = {"frozen": True}

    provider: str
    model: str
    usd_per_million_input: float = 0.0
    usd_per_million_output: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def cost_usd(self, usage: TokenUsage) -> float:
        return (
            usage.prompt_tokens * self.usd_per_million_input
            + usage.completion_tokens * self.usd_per_million_output
        ) / 1_000_000


class Completion(BaseModel):
    """A model response, with the accounting needed for a per-request cost meter."""

    model_config = {"frozen": True}

    text: str
    model: str = Field(description="The model that actually served this request.")
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 1
    fell_back: bool = Field(
        default=False,
        description="True when the first model in the chain did not serve this request.",
    )


class ChatProvider(Protocol):
    """
    Minimal provider contract.

    Implementations translate provider-specific transport failures into the gateway's
    error taxonomy; the gateway itself stays provider-agnostic.
    """

    name: str

    async def complete(
        self, model: str, messages: list[ChatMessage], *, temperature: float, max_tokens: int | None
    ) -> tuple[str, TokenUsage]:
        """Return (text, usage) or raise a GatewayError subclass."""
        ...
