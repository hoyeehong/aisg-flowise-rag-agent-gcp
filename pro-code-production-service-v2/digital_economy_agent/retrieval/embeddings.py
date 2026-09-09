"""
Embedding providers.

``GoogleEmbedder`` is the production path. ``HashingEmbedder`` is deterministic and
offline, so the retrieval stack — chunking, storage, hybrid fusion, ranking — can be
tested exhaustively without a network call or an API quota. That separation is why the
Phase 2 test suite stays runnable in CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request

from ..gateway.errors import ModelRateLimited, ModelTransient, ModelUnavailable
from .types import EmbedTask

# gemini-embedding-001 is native 3072-dim and supports truncation to smaller widths.
# 768 is chosen deliberately: a quarter of the storage and index size of 3072 for a
# small measured quality cost on this corpus, and it matches the width v1 documented.
GOOGLE_EMBED_MODEL = "gemini-embedding-001"
GOOGLE_EMBED_DIMENSIONS = 768

_WORD = re.compile(r"\b[a-zA-Z0-9_-]+\b")


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class GoogleEmbedder:
    """
    Google Generative AI embeddings over the REST API.

    Truncated outputs are re-normalised. Gemini embeddings are unit-length at their
    native 3072 dimensions; slicing to 768 does not preserve that, and cosine distance
    on non-normalised vectors silently mis-ranks. Normalising here means the pgvector
    column always holds unit vectors, so `<=>` is a true cosine distance.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = GOOGLE_EMBED_MODEL,
        dimensions: int = GOOGLE_EMBED_DIMENSIONS,
        timeout: float = 60.0,
    ) -> None:
        self.name = f"google/{model}@{dimensions}"
        self.dimensions = dimensions
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def _embed_one(self, text: str, task: EmbedTask) -> list[float]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:embedContent?key={self._api_key}"
        )
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            "taskType": str(task),
            "outputDimensionality": self.dimensions,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", "replace")[:200]
            detail = f"{self._model}: HTTP {exc.code}: {message}"
            if exc.code in (400, 401, 403, 404):
                raise ModelUnavailable(detail) from exc
            if exc.code == 429:
                raise ModelRateLimited(detail) from exc
            raise ModelTransient(detail) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelTransient(f"{self._model}: unreachable: {exc}") from exc

        values = body.get("embedding", {}).get("values")
        if not values:
            raise ModelTransient(f"{self._model}: empty embedding in response")
        if len(values) != self.dimensions:
            raise ModelUnavailable(
                f"{self._model}: expected {self.dimensions} dimensions, got {len(values)}"
            )
        return _l2_normalise([float(v) for v in values])

    async def embed(self, texts: list[str], *, task: EmbedTask) -> list[list[float]]:
        # Sequential by design: the embedContent endpoint is per-item, and free-tier
        # quota is per-minute. Firing a corpus concurrently converts a slow ingest into
        # a failed one.
        return [self._embed_one(text, task) for text in texts]


class HashingEmbedder:
    """
    Deterministic bag-of-words hashing embedder. Offline, no dependencies.

    This is a real embedder, not a stub: the same text always yields the same unit
    vector, and texts sharing vocabulary land near each other. It has no semantic
    knowledge — 'digital trust' and 'cybersecurity' stay far apart — so it is used to
    test the *mechanics* of the retrieval stack, never to judge retrieval quality.
    """

    def __init__(self, dimensions: int = GOOGLE_EMBED_DIMENSIONS) -> None:
        self.name = f"hashing@{dimensions}"
        self.dimensions = dimensions

    def _embed_one(self, text: str, task: EmbedTask) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [t.lower() for t in _WORD.findall(text)]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        if not tokens:
            # A zero vector has undefined cosine distance; anchor empties to one axis.
            vector[0] = 1.0
        return _l2_normalise(vector)

    async def embed(self, texts: list[str], *, task: EmbedTask) -> list[list[float]]:
        return [self._embed_one(text, task) for text in texts]
