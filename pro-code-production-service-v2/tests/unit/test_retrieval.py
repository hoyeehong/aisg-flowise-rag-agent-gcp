"""Retrieval unit tests: chunking, fusion and diversification, all offline."""

from __future__ import annotations

import math

import pytest

from digital_economy_agent.retrieval import (
    ChunkingConfig,
    EmbedTask,
    HashingEmbedder,
    ScoredChunk,
    chunk_pages,
    content_hash,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from digital_economy_agent.tools.types import Chunk


def _scored(pages: list[int], *, source: str = "a.pdf") -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(text=f"passage about topic {p}", source=source, page=p),
            score=1.0 / i,
            rank=i,
        )
        for i, p in enumerate(pages, start=1)
    ]


# --- chunking --------------------------------------------------------------


def test_chunks_never_span_pages():
    """A citation pointing at two pages is not checkable by a reader."""
    pages = [(1, "alpha. " * 200), (2, "beta. " * 200)]
    chunks = chunk_pages(pages, source="a.pdf", config=ChunkingConfig(chunk_size=200, overlap=40))
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert not (("alpha" in c.text) and ("beta" in c.text))


def test_chunks_overlap_so_context_spans_boundaries():
    pages = [(1, " ".join(f"word{i}" for i in range(400)))]
    chunks = chunk_pages(pages, source="a.pdf", config=ChunkingConfig(chunk_size=300, overlap=100))
    assert len(chunks) > 1
    # Consecutive chunks should share vocabulary because of the overlap window.
    first, second = set(chunks[0].text.split()), set(chunks[1].text.split())
    assert first & second


def test_short_page_is_a_single_chunk():
    chunks = chunk_pages(
        [(1, "a short but sufficiently long sentence about policy.")], source="a.pdf"
    )
    assert len(chunks) == 1


def test_empty_and_whitespace_pages_produce_nothing():
    assert chunk_pages([(1, ""), (2, "   \n  ")], source="a.pdf") == []


def test_overlap_must_be_smaller_than_chunk_size():
    """overlap >= chunk_size cannot make forward progress; reject it at construction."""
    with pytest.raises(ValueError, match="overlap"):
        ChunkingConfig(chunk_size=100, overlap=100)


def test_content_hash_distinguishes_page_and_source():
    """Identical boilerplate on two pages is two distinct citable chunks."""
    a = content_hash("a.pdf", 1, "same text")
    b = content_hash("a.pdf", 2, "same text")
    c = content_hash("b.pdf", 1, "same text")
    assert len({a, b, c}) == 3
    assert a == content_hash("a.pdf", 1, "same text")


# --- fusion ----------------------------------------------------------------


def test_rrf_rewards_agreement_between_strategies():
    """A chunk both strategies rank must beat one that only a single strategy ranks."""
    vector = _scored([10, 20, 30])
    lexical = _scored([30, 40, 50])
    fused = reciprocal_rank_fusion([(vector, 1.0), (lexical, 1.0)])
    assert fused[0][0].page == 30, "page 30 is the only one found by both"


def test_rrf_is_scale_free():
    """
    Ranks, not scores. Cosine similarity and ts_rank_cd are on different scales, so
    fusing raw scores would let whichever produces bigger numbers dominate.
    """
    small = [ScoredChunk(chunk=Chunk(text="s", source="a", page=1), score=0.001, rank=1)]
    large = [ScoredChunk(chunk=Chunk(text="l", source="a", page=2), score=999.0, rank=1)]
    fused = reciprocal_rank_fusion([(small, 1.0), (large, 1.0)])
    assert fused[0][1] == pytest.approx(fused[1][1]), "equal ranks must fuse equally"


def test_rrf_weights_shift_the_balance():
    vector = _scored([1, 2])
    lexical = _scored([2, 1])
    lexical_led = reciprocal_rank_fusion([(vector, 0.1), (lexical, 10.0)])
    assert lexical_led[0][0].page == 2


def test_rrf_on_empty_input_is_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([([], 1.0)]) == []


# --- diversification -------------------------------------------------------


def test_mmr_drops_near_duplicates():
    """
    Adjacent chunks share text because of the 200-char overlap, so an un-diversified
    top-k can spend several slots on the same passage.
    """
    shared = "digital trust cybersecurity responsible artificial intelligence frameworks"
    candidates = [
        (Chunk(text=shared, source="a", page=1), 1.00),
        (Chunk(text=shared, source="a", page=2), 0.99),
        (
            Chunk(text="talent pipeline shortages constrain regional hiring", source="a", page=3),
            0.50,
        ),
    ]
    picked = maximal_marginal_relevance(candidates, top_k=2, lambda_=0.5)
    pages = [c.page for c, _ in picked]
    assert pages == [1, 3], "the duplicate of page 1 must lose to the distinct passage"


def test_mmr_lambda_one_is_pure_relevance():
    shared = "identical text here for both entries"
    candidates = [
        (Chunk(text=shared, source="a", page=1), 1.0),
        (Chunk(text=shared, source="a", page=2), 0.9),
        (Chunk(text="something entirely different", source="a", page=3), 0.1),
    ]
    picked = maximal_marginal_relevance(candidates, top_k=2, lambda_=1.0)
    assert [c.page for c, _ in picked] == [1, 2]


def test_mmr_on_empty_input_is_empty():
    assert maximal_marginal_relevance([], top_k=5, lambda_=0.7) == []


# --- embeddings ------------------------------------------------------------


async def test_hashing_embedder_is_deterministic_and_unit_length():
    emb = HashingEmbedder(dimensions=128)
    a, b = await emb.embed(["digital trust", "digital trust"], task=EmbedTask.DOCUMENT)
    assert a == b
    assert math.isclose(sum(v * v for v in a), 1.0, abs_tol=1e-9)
    assert emb.dimensions == 128


async def test_hashing_embedder_handles_empty_text():
    """A zero vector has undefined cosine distance, so empties must still be unit-length."""
    (v,) = await HashingEmbedder(dimensions=64).embed([""], task=EmbedTask.DOCUMENT)
    assert math.isclose(sum(x * x for x in v), 1.0, abs_tol=1e-9)


async def test_shared_vocabulary_scores_higher_than_none():
    emb = HashingEmbedder(dimensions=256)
    base, near, far = await emb.embed(
        [
            "digital trust and cybersecurity",
            "digital trust frameworks",
            "unrelated maritime shipping tariffs",
        ],
        task=EmbedTask.DOCUMENT,
    )
    dot = lambda x, y: sum(a * b for a, b in zip(x, y, strict=True))  # noqa: E731
    assert dot(base, near) > dot(base, far)
