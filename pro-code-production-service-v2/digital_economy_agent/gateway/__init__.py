"""Model gateway: provider-agnostic routing with retries, fallback and cost metering."""

from .errors import (
    AllModelsFailed,
    GatewayError,
    ModelRateLimited,
    ModelTransient,
    ModelUnavailable,
)
from .gateway import GatewayStats, ModelGateway, RetryPolicy
from .providers import OpenAICompatibleProvider
from .types import ChatMessage, ChatProvider, Completion, ModelSpec, TokenUsage

__all__ = [
    "AllModelsFailed",
    "ChatMessage",
    "ChatProvider",
    "Completion",
    "GatewayError",
    "GatewayStats",
    "ModelGateway",
    "ModelRateLimited",
    "ModelSpec",
    "ModelTransient",
    "ModelUnavailable",
    "OpenAICompatibleProvider",
    "RetryPolicy",
    "TokenUsage",
]
