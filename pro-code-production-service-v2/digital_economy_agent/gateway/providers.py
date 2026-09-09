"""Chat providers. Each maps transport failures onto the gateway error taxonomy."""

from __future__ import annotations

import json

import httpx

from .errors import ModelRateLimited, ModelTransient, ModelUnavailable
from .types import ChatMessage, TokenUsage

# Permanent for the run: wrong model, malformed request, or missing/denied credential.
_PERMANENT_STATUS = frozenset({400, 401, 403, 404})


def _retry_after(response: httpx.Response) -> float | None:
    """Read a retry hint from the standard header or a google.rpc.RetryInfo detail."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = response.json().get("error", {}).get("details", [])
    except (json.JSONDecodeError, AttributeError, ValueError):
        return None
    for detail in details if isinstance(details, list) else []:
        if "RetryInfo" in str(detail.get("@type", "")):
            try:
                return float(str(detail.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                return None
    return None


def _message(response: httpx.Response, limit: int = 200) -> str:
    """Extract a readable message from a provider error envelope."""
    try:
        body = response.json()
        text = body.get("error", {}).get("message") or json.dumps(body)
    except (json.JSONDecodeError, ValueError):
        text = response.text
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _raise_for_status(response: httpx.Response, model: str) -> None:
    if response.is_success:
        return
    detail = f"{model}: HTTP {response.status_code}: {_message(response)}"
    if response.status_code in _PERMANENT_STATUS:
        raise ModelUnavailable(detail)
    if response.status_code == 429:
        raise ModelRateLimited(detail, retry_after=_retry_after(response))
    raise ModelTransient(detail, retry_after=_retry_after(response))


class OpenAICompatibleProvider:
    """
    Provider for any OpenAI-compatible ``/chat/completions`` endpoint.

    One implementation covers Groq (the v1 system under test), OpenAI, vLLM and TGI,
    which is the practical reason to prefer this wire format for self-hosted models.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    async def _post(self, payload: dict[str, object]) -> httpx.Response:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, json=payload, headers=headers)

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> tuple[str, TokenUsage]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = await self._post(payload)
        except httpx.TimeoutException as exc:
            raise ModelTransient(f"{model}: timeout after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise ModelTransient(f"{model}: transport error: {exc}") from exc

        _raise_for_status(response, model)

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
            # A malformed or empty choice is a property of this response, not the model,
            # so it is transient and worth one more attempt.
            raise ModelTransient(f"{model}: unparseable response: {response.text[:200]}") from exc

        raw_usage = body.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
        )
        if text is None:
            raise ModelTransient(f"{model}: null completion content")
        return text, usage
