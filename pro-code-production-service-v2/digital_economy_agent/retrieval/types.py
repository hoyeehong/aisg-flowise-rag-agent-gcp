"""Embedding contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class EmbedTask(StrEnum):
    """
    Asymmetric embedding task types.

    A document and a query about that document are different kinds of text, and models
    that accept a task hint place them in the same neighbourhood more reliably than a
    single symmetric embedding does. Using the wrong side is a silent recall loss: the
    vectors are still valid, just less well aligned.
    """

    DOCUMENT = "RETRIEVAL_DOCUMENT"
    QUERY = "RETRIEVAL_QUERY"


class Embedder(Protocol):
    """
    Text-to-vector seam.

    ``dimensions`` must match the pgvector column width, so it is part of the contract
    rather than an implementation detail: a mismatch is a runtime insert failure.
    """

    name: str
    dimensions: int

    async def embed(self, texts: list[str], *, task: EmbedTask) -> list[list[float]]: ...
