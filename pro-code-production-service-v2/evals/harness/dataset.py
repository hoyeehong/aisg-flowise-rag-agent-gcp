"""Golden-set loading and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .types import GoldenCase

GOLDEN_DIR = Path(__file__).parent.parent / "golden"


class DatasetError(ValueError):
    """A golden set is malformed. Always fatal: a partial dataset skews every mean."""


def load_dataset(path: Path) -> list[GoldenCase]:
    """
    Load one ``.jsonl`` golden set.

    A malformed line is fatal rather than skipped. Silently dropping cases changes the
    denominator of every aggregate, so a score would no longer be comparable to a
    baseline computed over the full set.
    """
    if not path.is_file():
        raise DatasetError(f"no such golden set: {path}")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate_json(line)
        except Exception as exc:
            raise DatasetError(f"{path.name}:{number}: {exc}") from exc
        if case.case_id in seen:
            raise DatasetError(f"{path.name}:{number}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{path.name}: contains no cases")
    return cases


def load_datasets(paths: list[Path]) -> list[GoldenCase]:
    """Load several golden sets, rejecting case_id collisions across files."""
    combined: list[GoldenCase] = []
    seen: dict[str, str] = {}
    for path in paths:
        for case in load_dataset(path):
            if case.case_id in seen:
                raise DatasetError(
                    f"case_id {case.case_id!r} appears in both {seen[case.case_id]} and {path.name}"
                )
            seen[case.case_id] = path.name
            combined.append(case)
    return combined


def discover_datasets(directory: Path = GOLDEN_DIR) -> list[Path]:
    return sorted(directory.glob("*.jsonl"))


def dataset_fingerprint(paths: list[Path]) -> str:
    """
    One hash over every dataset file, recorded in the report.

    A score is only comparable to a baseline computed over the same cases, so the
    fingerprint is what makes "this regressed" a defensible claim rather than a guess.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
