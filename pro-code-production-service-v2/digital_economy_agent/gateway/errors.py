"""
Model gateway error taxonomy.

The three classes are not interchangeable, and conflating them is the most common way
a retry policy wastes time:

* ``ModelRateLimited`` — quota is gone for now. It does not come back in seconds, so
  sleeping on it stalls the caller while a sibling model is very likely free. Advance.
* ``ModelTransient`` — a server fault or timeout. These genuinely clear in seconds, so
  a bounded backoff against the *same* model is worthwhile.
* ``ModelUnavailable`` — the model name is wrong or the credential lacks access. This
  never succeeds, so the model must be retired rather than retried per request.

This taxonomy was derived empirically while fixing the v1 evaluation judge: an earlier
implementation slept 42s on a 429 before discovering the wait was futile.
"""

from __future__ import annotations


class GatewayError(RuntimeError):
    """Base class for every model gateway failure."""


class ModelUnavailable(GatewayError):
    """Permanent for this run: unknown model, bad request, or missing access."""


class ModelRateLimited(GatewayError):
    """Quota exhausted. Carries the provider's suggested reset delay when supplied."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelTransient(GatewayError):
    """A transient server fault or timeout that typically clears in seconds."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AllModelsFailed(GatewayError):
    """
    Every model in the chain was tried and none produced a completion.

    ``categories`` records *why* each model failed, so a caller can distinguish a
    transient capacity problem the client should retry from a misconfiguration it
    never should. Without it the HTTP layer can only guess, and guessing means
    returning 500 for a dependency failure the service did not cause.
    """

    def __init__(
        self,
        message: str,
        attempts: dict[str, str],
        *,
        categories: dict[str, str] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.categories = categories or {}
        self.retry_after = retry_after

    @property
    def rate_limited_only(self) -> bool:
        """True when every model failed purely on quota — worth retrying later."""
        return bool(self.categories) and all(c == "rate_limited" for c in self.categories.values())
