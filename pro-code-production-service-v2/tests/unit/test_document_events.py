"""
Document event parsing and validation.

Parsing failures must be *permanent*: a payload that is not a document event will not
become one on redelivery, and retrying it five times delays every message behind it to
reach a conclusion already available on the first attempt.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from digital_economy_agent.messaging import (
    DocumentIngestionEvent,
    DocumentIngestionHandler,
    IncomingMessage,
    PermanentMessageError,
)


def _handler() -> DocumentIngestionHandler:
    # The store and embedder are never reached: every case here fails during parsing,
    # which is the property being tested.
    return DocumentIngestionHandler(cast(Any, None), cast(Any, None))


def _message(payload: Any, *, raw: bytes | None = None) -> IncomingMessage:
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return IncomingMessage(id="m1", data=data)


VALID = {
    "tenant_id": "acme",
    "source": "report.pdf",
    "pages": [{"page": 1, "text": "Digital trust and governance."}],
}


def test_a_valid_event_parses() -> None:
    event = DocumentIngestionEvent.model_validate(VALID)
    assert event.tenant_id == "acme"
    assert event.pages[0].page == 1
    assert event.source_updated_at is None


def test_source_updated_at_is_parsed_when_present() -> None:
    event = DocumentIngestionEvent.model_validate(
        {**VALID, "source_updated_at": "2026-03-01T10:00:00Z"}
    )
    assert event.source_updated_at == datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


async def test_non_json_is_permanent() -> None:
    with pytest.raises(PermanentMessageError, match="not valid JSON"):
        await _handler()(_message(None, raw=b"{not json"))


async def test_undecodable_bytes_are_permanent() -> None:
    with pytest.raises(PermanentMessageError, match="not valid JSON"):
        await _handler()(_message(None, raw=b"\xff\xfe\x00binary"))


@pytest.mark.parametrize(
    "payload,label",
    [
        ({}, "empty"),
        ({"source": "r.pdf", "pages": [{"page": 1, "text": "x"}]}, "no tenant"),
        ({"tenant_id": "acme", "pages": [{"page": 1, "text": "x"}]}, "no source"),
        ({"tenant_id": "acme", "source": "r.pdf"}, "no pages"),
        ({"tenant_id": "acme", "source": "r.pdf", "pages": []}, "empty pages"),
        ({"tenant_id": "acme", "source": "", "pages": [{"page": 1, "text": "x"}]}, "blank source"),
        (
            {"tenant_id": "acme", "source": "r.pdf", "pages": [{"page": 0, "text": "x"}]},
            "page zero",
        ),
        ({"tenant_id": "acme", "source": "r.pdf", "pages": "not a list"}, "pages wrong type"),
        (["not", "an", "object"], "top-level array"),
    ],
)
async def test_structurally_invalid_events_are_permanent(payload: Any, label: str) -> None:
    with pytest.raises(PermanentMessageError, match="not a document event"):
        await _handler()(_message(payload))


@pytest.mark.parametrize(
    "tenant",
    ["", "  ", "../other", "tenant with spaces", "a" * 65, "-leading", "acme/../root"],
)
async def test_a_malformed_tenant_is_permanent(tenant: str) -> None:
    """
    The same tenant rule as the HTTP path, shared so the two cannot drift.

    A tenant id becomes a Postgres setting that the RLS policy compares against, so a
    malformed one either matches nothing -- indistinguishable from an empty corpus --
    or resolves somewhere it should not.
    """
    with pytest.raises(PermanentMessageError, match="not a document event"):
        await _handler()(_message({**VALID, "tenant_id": tenant}))


def test_the_tenant_rule_matches_the_http_path() -> None:
    """Both paths must accept and reject exactly the same identifiers."""
    from digital_economy_agent.tenancy import is_valid_tenant_id

    for good in ("acme", "acme-corp", "tenant.1", "A_b-c.9", "x" * 64):
        assert is_valid_tenant_id(good)
        assert DocumentIngestionEvent.model_validate({**VALID, "tenant_id": good})
    for bad in ("", "-x", ".x", "_x", "x" * 65, "a b", "a/b"):
        assert not is_valid_tenant_id(bad)
        with pytest.raises(ValueError):
            DocumentIngestionEvent.model_validate({**VALID, "tenant_id": bad})
