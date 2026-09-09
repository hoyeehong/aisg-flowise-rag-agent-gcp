"""Retrieval: chunking, embeddings, pgvector storage and hybrid search."""

from .chunking import ChunkingConfig, chunk_pages
from .embeddings import (
    GOOGLE_EMBED_DIMENSIONS,
    GOOGLE_EMBED_MODEL,
    GoogleEmbedder,
    HashingEmbedder,
)
from .hybrid import (
    RRF_K,
    HybridConfig,
    HybridRetriever,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from .store import PgVectorStore, ScoredChunk, content_hash
from .types import Embedder, EmbedTask

__all__ = [
    "GOOGLE_EMBED_DIMENSIONS",
    "GOOGLE_EMBED_MODEL",
    "RRF_K",
    "ChunkingConfig",
    "EmbedTask",
    "Embedder",
    "GoogleEmbedder",
    "HashingEmbedder",
    "HybridConfig",
    "HybridRetriever",
    "PgVectorStore",
    "ScoredChunk",
    "chunk_pages",
    "content_hash",
    "maximal_marginal_relevance",
    "reciprocal_rank_fusion",
]
