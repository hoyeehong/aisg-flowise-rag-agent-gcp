"""
Redaction tests.

The governing principle is asymmetric: a false positive costs a redacted word, a false
negative is a data breach. These tests therefore assert that detection happens even
when a checksum fails, and that a checksum failure is *reported* rather than acted on.
"""

from __future__ import annotations

import pytest

from digital_economy_agent.ingestion import (
    NullRedactor,
    PatternRedactor,
    luhn_valid,
    nric_checksum_valid,
)


@pytest.fixture
def redactor() -> PatternRedactor:
    return PatternRedactor()


@pytest.mark.parametrize(
    ("text", "label", "residue"),
    [
        ("mail me at analyst@ncs.com.sg please", "email", "analyst"),
        ("my NRIC is S1234567D ok", "nric_fin", "S1234567D"),
        ("card 4111 1111 1111 1111 charged", "credit_card", "4111"),
        ("call +65 9123 4567 now", "phone", "9123"),
        ("host at 10.20.30.40 responds", "ip_address", "10.20"),
    ],
)
def test_each_identifier_type_is_removed(redactor, text, label, residue):
    out, report = redactor.redact(text)
    assert report.counts.get(label) == 1
    assert residue not in out, "the original identifier must not survive"


def test_policy_prose_is_untouched(redactor):
    """Over-redaction is preferred, but not at the cost of mangling the corpus."""
    text = (
        "ASEAN DEFA harmonises cross-border data flows. In 2024 the SEA-6 economies "
        "reported growth of 15 percent across digital services."
    )
    out, report = redactor.redact(text)
    assert out == text
    assert report.total == 0


def test_checksum_failure_still_redacts_but_is_flagged(redactor):
    """
    A checksum is a confidence signal, never a veto.

    If it were a veto, a bug in the checksum implementation would convert a detection
    into a silent leak -- the one failure mode this module must not have.
    """
    out, report = redactor.redact("id S1234567Z here")  # Z is the wrong check letter
    assert "S1234567Z" not in out
    assert report.counts["nric_fin"] == 1
    assert report.low_confidence["nric_fin"] == 1


def test_multiple_identifiers_are_all_counted(redactor):
    out, report = redactor.redact("a@b.co and c@d.co, NRIC S1234567D, phone 91234567")
    assert report.counts["email"] == 2
    assert report.counts["nric_fin"] == 1
    assert report.counts["phone"] == 1
    assert report.total == 4
    assert "@" not in out


def test_null_redactor_is_a_deliberate_opt_out():
    text = "email a@b.co"
    out, report = NullRedactor().redact(text)
    assert out == text
    assert report.total == 0


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("4111111111111111", True),  # canonical Visa test number
        ("4111111111111112", False),  # last digit altered
        ("79927398713", True),  # canonical Luhn vector, shorter than a card
        ("4111-1111 1111 1111", True),  # separators ignored
        ("", False),
    ],
)
def test_luhn(number, valid):
    assert luhn_valid(number) is valid


@pytest.mark.parametrize(
    ("nric", "valid"),
    [
        ("S1234567D", True),  # externally known-valid NRIC: the correctness anchor
        ("s1234567d", True),  # case-insensitive
        ("S1234567Z", False),  # wrong check letter
        ("T0000001E", True),  # T prefix applies a +4 offset
        ("T0000001B", False),
        ("not-an-nric", False),
        ("S123456", False),  # too short
        ("X1234567D", False),  # unknown prefix
    ],
)
def test_nric_checksum(nric, valid):
    assert nric_checksum_valid(nric) is valid


@pytest.mark.parametrize("prefix", ["S", "T", "F", "G", "M"])
def test_exactly_one_check_letter_validates_per_number(prefix):
    """
    Property check: for any 7-digit body there must be exactly one valid check letter.

    This catches the implementation bugs a handful of hand-picked vectors would miss --
    a wrong weight, a shifted offset, or a mistyped lookup table would almost always
    break the one-letter-per-number property somewhere.
    """
    import string

    for body in ("0000001", "1234567", "9876543", "0000000"):
        valid = [c for c in string.ascii_uppercase if nric_checksum_valid(f"{prefix}{body}{c}")]
        assert len(valid) == 1, f"{prefix}{body}: expected 1 valid letter, got {valid}"
