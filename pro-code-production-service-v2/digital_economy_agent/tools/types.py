"""
Typed tool contracts.

Every tool the agent can call is declared here as a Pydantic-validated input/output
pair plus a Protocol. v1 shipped ``agentTools: []`` — the agent had no tools at all and
retrieval was an implicit binding to Flowise server state. Declaring the contract in
code is what makes a tool unit-testable and its failures attributable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrieved passage, carrying the metadata needed to cite it."""

    model_config = {"frozen": True}

    text: str
    source: str = Field(description="Document identifier, e.g. the source PDF name.")
    page: int | None = None
    score: float | None = Field(default=None, description="Retriever similarity score.")

    def citation(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page is not None else self.source


class RetrievalRequest(BaseModel):
    """Input contract for a retrieval tool."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    # Empty means "whatever tenant this retriever was constructed for", which is the
    # single-tenant path and what the in-memory retriever always does. A request that
    # names a tenant has had it resolved from a verified credential, never from the
    # request body -- see api/auth.py.
    tenant_id: str = ""


class RetrievalResult(BaseModel):
    """
    Output contract for a retrieval tool.

    ``chunks`` may legitimately be empty. Callers must handle that rather than assume
    a non-empty context: an agent that writes a report from zero chunks is the exact
    failure mode the v1 groundedness measurement could not detect.
    """

    chunks: list[Chunk] = []
    query: str = ""

    @property
    def context(self) -> str:
        """Concatenated context, each passage prefixed with its citation."""
        return "\n\n".join(f"[{c.citation()}] {c.text}" for c in self.chunks)

    @property
    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)


@runtime_checkable
class Retriever(Protocol):
    """
    Retrieval seam.

    Phase 1 ships an in-memory implementation so the graph and API are testable without
    infrastructure. Phase 2 substitutes a pgvector-backed hybrid retriever behind this
    same Protocol; nothing in ``agents/`` or ``api/`` changes when it does.
    """

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...
