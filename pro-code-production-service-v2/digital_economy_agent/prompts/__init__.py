"""
Prompt loading with pinned versions.

Prompts are versioned in the *filename* (``<node>.v<n>.md``), not only in git history.
Two properties follow that git alone does not give:

* A rollback is a one-line change to ``PINNED``, reviewable in isolation.
* An evaluation report can cite the exact prompt version that produced a score, which
  makes the score attributable.

Released prompt files are immutable. A changed prompt is a new ``.v<n>`` file plus a
bump here, so an old score always remains reproducible.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent

# The prompt version each node runs. Bump deliberately, alongside an eval run.
PINNED: dict[str, int] = {
    "research": 1,
    "write_draft": 1,
    "revise": 1,
}


@cache
def load(node: str, version: int | None = None) -> str:
    """Return the pinned (or explicitly requested) prompt template for a node."""
    if node not in PINNED:
        raise KeyError(f"unknown prompt node {node!r}; known: {', '.join(sorted(PINNED))}")
    resolved = PINNED[node] if version is None else version
    path = _DIR / f"{node}.v{resolved}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt file missing: {path.name}")
    return path.read_text(encoding="utf-8").strip()


def pinned_versions() -> dict[str, str]:
    """Prompt identifiers for provenance, e.g. {'research': 'research.v1'}."""
    return {node: f"{node}.v{version}" for node, version in sorted(PINNED.items())}
