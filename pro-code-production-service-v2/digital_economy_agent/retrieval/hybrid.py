"""
Hybrid retrieval: vector + lexical search fused by Reciprocal Rank Fusion, with
optional query rewriting and MMR diversification.

Why hybrid rather than pure vector search: the two strategies fail in opposite
directions. Embeddings generalise but blur exact tokens, so a query for ``ASEAN DEFA``
can rank a passage about "regional cooperation" above the one naming the agreement.
Lexical search nails the identifier but cannot connect "Tech for Good" to "digital
trust" at all — the measured failure of v1's token-overlap retriever, which matched
1 of 4 chunks on a well-formed query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..gateway import ChatMessage, ModelGateway
from ..gateway.errors import GatewayError
from ..tools.types import Chunk, RetrievalRequest, RetrievalResult
from .store import PgVectorStore, ScoredChunk
from .types import Embedder, EmbedTask

logger = logging.getLogger(__name__)

# RRF's damping constant. 60 is the value from the original Cormack et al. formulation
# and is deliberately not tuned here: with a 40-case golden set (Phase 3) it can be
# fitted, but tuning it against a handful of ad-hoc queries would just be overfitting.
RRF_K = 60

_REWRITE_PROMPT = """Rewrite the search query below into {n} alternative queries that \
would retrieve relevant passages from a policy report. Vary the vocabulary: include \
domain synonyms and specific named concepts the original query implies but does not \
state.

Return one query per line, no numbering, no commentary.

QUERY: {query}"""


@dataclass(frozen=True)
class HybridConfig:
    """Retrieval parameters, recorded in provenance so a result is reproducible."""

    # Each strategy fetches deeper than the final top_k: fusion can only reorder what
    # it is given, so a candidate missing from both pools can never be recovered.
    candidate_multiplier: int = 4
    min_candidates: int = 20
    vector_weight: float = 1.0
    lexical_weight: float = 1.0
    # MMR trade-off: 1.0 is pure relevance, 0.0 pure diversity.
    mmr_lambda: float = 0.7
    enable_mmr: bool = True
    rewrite_variants: int = 0  # 0 disables LLM query rewriting


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[ScoredChunk], float]], *, k: int = RRF_K
) -> list[tuple[Chunk, float]]:
    """
    Fuse ranked lists by RRF: ``score = sum(weight / (k + rank))``.

    Rank-based rather than score-based on purpose. Cosine similarity and ``ts_rank_cd``
    are not on comparable scales, so normalising and adding them would let whichever
    strategy happens to produce larger numbers dominate. Ranks are scale-free.
    """
    fused: dict[tuple[str, int | None, str], tuple[Chunk, float]] = {}
    for ranked, weight in ranked_lists:
        for item in ranked:
            key = (item.chunk.source, item.chunk.page, item.chunk.text[:120])
            contribution = weight / (k + item.rank)
            if key in fused:
                chunk, total = fused[key]
                fused[key] = (chunk, total + contribution)
            else:
                fused[key] = (item.chunk, contribution)
    return sorted(fused.values(), key=lambda pair: pair[1], reverse=True)


def _tokens(text: str) -> set[str]:
    return {w for w in text.lower().split() if len(w) > 3}


def maximal_marginal_relevance(
    candidates: list[tuple[Chunk, float]], *, top_k: int, lambda_: float
) -> list[tuple[Chunk, float]]:
    """
    Greedy MMR selection over lexical overlap.

    Overlapping chunks are a real problem here rather than a theoretical one: a 200-char
    overlap between adjacent chunks means neighbours share text, so an un-diversified
    top-5 can spend three slots on the same passage and starve the answer of breadth.

    This is a diversification pass, not a cross-encoder re-ranker; a cross-encoder that
    scores query-passage pairs directly is a later upgrade.
    """
    if not candidates:
        return []
    selected: list[tuple[Chunk, float]] = [candidates[0]]
    remaining = list(candidates[1:])

    while remaining and len(selected) < top_k:
        best_index, best_value = 0, float("-inf")
        for index, (chunk, relevance) in enumerate(remaining):
            chunk_tokens = _tokens(chunk.text)
            redundancy = 0.0
            for chosen, _ in selected:
                chosen_tokens = _tokens(chosen.text)
                union = chunk_tokens | chosen_tokens
                if union:
                    redundancy = max(redundancy, len(chunk_tokens & chosen_tokens) / len(union))
            value = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        selected.append(remaining.pop(best_index))

    return selected


class HybridRetriever:
    """
    Implements the ``Retriever`` protocol, so it drops into the agent graph unchanged.

    Nothing in ``agents/`` or ``api/`` knows this replaced the in-memory retriever —
    that was the point of declaring the protocol in Phase 1.
    """

    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        *,
        config: HybridConfig | None = None,
        gateway: ModelGateway | None = None,
        tenant_id: str = "default",
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or HybridConfig()
        self._gateway = gateway
        self._tenant_id = tenant_id

    def describe(self) -> str:
        parts = [f"hybrid(vector+lexical, RRF k={RRF_K})", f"embedder={self._embedder.name}"]
        if self._config.enable_mmr:
            parts.append(f"mmr(lambda={self._config.mmr_lambda})")
        if self._config.rewrite_variants:
            parts.append(f"rewrite x{self._config.rewrite_variants}")
        return " ".join(parts)

    async def _rewrite(self, query: str) -> list[str]:
        """Expand the query into variants. Failure degrades to the original query."""
        n = self._config.rewrite_variants
        if not n or self._gateway is None:
            return [query]
        try:
            completion = await self._gateway.complete(
                [ChatMessage(role="user", content=_REWRITE_PROMPT.format(n=n, query=query))],
                temperature=0.3,
            )
        except GatewayError as exc:
            # Rewriting is an enhancement. Losing it must not fail the retrieval.
            logger.warning("query rewriting unavailable, using original query: %s", exc)
            return [query]

        variants = [line.strip(" -•\t") for line in completion.text.splitlines()]
        return [query, *[v for v in variants if v][:n]]

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        cfg = self._config
        depth = max(request.top_k * cfg.candidate_multiplier, cfg.min_candidates)
        queries = await self._rewrite(request.query)

        ranked_lists: list[tuple[list[ScoredChunk], float]] = []
        for query in queries:
            embedding = (await self._embedder.embed([query], task=EmbedTask.QUERY))[0]
            ranked_lists.append(
                (
                    await self._store.vector_search(
                        embedding, top_k=depth, tenant_id=self._tenant_id
                    ),
                    cfg.vector_weight,
                )
            )
            ranked_lists.append(
                (
                    await self._store.lexical_search(query, top_k=depth, tenant_id=self._tenant_id),
                    cfg.lexical_weight,
                )
            )

        fused = reciprocal_rank_fusion(ranked_lists)
        if cfg.enable_mmr:
            fused = maximal_marginal_relevance(fused, top_k=request.top_k, lambda_=cfg.mmr_lambda)

        chunks = [
            chunk.model_copy(update={"score": round(score, 6)})
            for chunk, score in fused[: request.top_k]
        ]
        return RetrievalResult(chunks=chunks, query=request.query)
