"""
In-memory retriever: a real implementation of the Retriever protocol over a fixed
corpus, using token-overlap scoring.

This is deliberately not a vector search. It exists so the graph, the API and the
tests can run end to end with no database, and so Phase 2 has a behavioural baseline
to compare a pgvector implementation against. It is not a production retriever, and
``describe()`` says so at runtime.
"""

from __future__ import annotations

import re

from .types import Chunk, RetrievalRequest, RetrievalResult

_WORD = re.compile(r"\b[a-zA-Z0-9_-]+\b")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
    ]
)


def _tokens(text: str) -> set[str]:
    return {t for t in (w.lower() for w in _WORD.findall(text)) if t not in _STOPWORDS}


class InMemoryRetriever:
    """Scores chunks by Jaccard-style token overlap with the query."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._index = [(c, _tokens(c.text)) for c in self._chunks]

    @staticmethod
    def describe() -> str:
        return "in-memory token-overlap retriever (development only; not a vector search)"

    async def corpus_coverage(self, tenant_id: str = "") -> dict[str, list[int]]:
        """
        Indexed pages per source, for evaluation preconditions.

        ``tenant_id`` is accepted and ignored: this retriever holds a single in-process
        corpus and has no tenancy at all. Matching the signature keeps the two
        retrievers interchangeable behind one route, and the argument being ignored
        here is precisely why this retriever is a development and test fixture rather
        than something to serve multiple tenants from.
        """
        pages: dict[str, set[int]] = {}
        for chunk in self._chunks:
            if chunk.page is not None:
                pages.setdefault(chunk.source, set()).add(chunk.page)
        return {source: sorted(p) for source, p in pages.items()}

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        query_tokens = _tokens(request.query)
        if not query_tokens:
            return RetrievalResult(chunks=[], query=request.query)

        scored: list[tuple[float, Chunk]] = []
        for chunk, tokens in self._index:
            if not tokens:
                continue
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens | tokens)
            scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [
            chunk.model_copy(update={"score": round(score, 4)})
            for score, chunk in scored[: request.top_k]
        ]
        return RetrievalResult(chunks=top, query=request.query)
