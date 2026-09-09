"""
Page-aware recursive chunking.

v1 configured chunk size 1000 / overlap 200 inside the Flowise UI, so the settings that
determined its retrieval quality were unreproducible from the repository. Declaring the
strategy here is the point: it can be diffed, tested, and cited by an eval report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..tools.types import Chunk

# Split at the largest natural boundary that fits, falling back to progressively finer
# ones. Paragraph breaks preserve the most meaning; a hard character cut preserves none,
# so it is only ever the last resort.
_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters. Recorded in ingestion provenance so a run is reproducible."""

    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_chars: int = 60

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.overlap < self.chunk_size:
            # overlap >= chunk_size cannot make progress and would loop forever.
            raise ValueError("overlap must be non-negative and smaller than chunk_size")


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    """Recursively split one page's text into overlapping windows."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= config.chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + config.chunk_size
        if end >= len(text):
            pieces.append(text[start:].strip())
            break

        # Prefer the latest separator inside the window, so the cut lands on a boundary.
        cut = -1
        for separator in _SEPARATORS:
            found = text.rfind(separator, start + config.min_chunk_chars, end)
            if found != -1:
                cut = found + len(separator)
                break
        if cut == -1:
            cut = end

        pieces.append(text[start:cut].strip())
        # Step back by the overlap so context spans the boundary; max() guarantees
        # forward progress even when the overlap would otherwise stall it.
        start = max(cut - config.overlap, start + 1)

    return [p for p in pieces if len(p) >= config.min_chunk_chars]


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    source: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """
    Chunk ``(page_number, text)`` pairs, keeping each chunk attributable to its page.

    Chunks never span pages. A citation that points at two pages at once is not
    checkable by a reader, which defeats the purpose of carrying citations at all.
    """
    resolved = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for page_number, page_text in pages:
        for piece in _split_text(page_text, resolved):
            chunks.append(Chunk(text=piece, source=source, page=page_number))
    return chunks
