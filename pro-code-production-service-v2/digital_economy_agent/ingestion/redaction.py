"""
PII redaction for the ingestion pipeline.

DESIGN PRINCIPLE: in redaction, false positives are cheap and false negatives are a
data breach. Every rule here therefore errs toward over-redaction. Checksums (Luhn,
NRIC) are used only to *raise confidence* and annotate a finding — never to reject a
pattern match, because a bug in a checksum implementation would silently turn a
detection into a leak.

Scope: a ``Redactor`` protocol plus a dependency-free pattern implementation covering
the identifier types that appear in Singapore/SEA enterprise data. Microsoft Presidio
is a drop-in alternative behind the same protocol and adds NER-based detection of names
and locations; it is deliberately not the default because it pulls spaCy plus a language
model, which would dominate both CI time and the runtime image for detection classes
this corpus does not contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# NRIC/FIN check letters. Used for confidence annotation only -- see the module note.
_NRIC_WEIGHTS = (2, 7, 6, 5, 4, 3, 2)
_NRIC_ST_LETTERS = "JZIHGFEDCBA"
_NRIC_FG_LETTERS = "XWUTRQPNMLK"
_NRIC_OFFSETS = {"S": 0, "T": 4, "F": 0, "G": 4, "M": 3}


@dataclass(frozen=True)
class RedactionRule:
    """One detector: a pattern, the token it is replaced with, and a label."""

    label: str
    pattern: re.Pattern[str]
    placeholder: str


@dataclass
class RedactionReport:
    """What was removed, for metadata governance and audit."""

    counts: dict[str, int] = field(default_factory=dict)
    low_confidence: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: RedactionReport) -> None:
        for label, n in other.counts.items():
            self.counts[label] = self.counts.get(label, 0) + n
        for label, n in other.low_confidence.items():
            self.low_confidence[label] = self.low_confidence.get(label, 0) + n


class Redactor(Protocol):
    """Redaction seam, so Presidio or a cloud DLP service can be substituted."""

    name: str

    def redact(self, text: str) -> tuple[str, RedactionReport]: ...


def luhn_valid(digits: str) -> bool:
    """
    Pure Luhn checksum. Separators are ignored.

    Deliberately carries no length rule: card length is a property of the *pattern*
    (see the credit_card rule) and mixing it in here made a function named for a
    checksum silently reject valid Luhn strings that happened to be short.
    """
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 2:
        return False
    total = 0
    for index, digit in enumerate(reversed(numbers)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def nric_checksum_valid(value: str) -> bool:
    """
    Singapore NRIC/FIN check letter.

    Confidence signal only. A ``False`` result never prevents redaction: an
    NRIC-shaped string is redacted regardless, because the cost of being wrong in
    the other direction is leaking a national identifier.
    """
    value = value.upper().strip()
    if len(value) != 9 or value[0] not in _NRIC_OFFSETS or not value[1:8].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(value[1:8], _NRIC_WEIGHTS, strict=True))
    total += _NRIC_OFFSETS[value[0]]
    table = _NRIC_ST_LETTERS if value[0] in ("S", "T") else _NRIC_FG_LETTERS
    return value[8] == table[total % 11]


DEFAULT_RULES: tuple[RedactionRule, ...] = (
    RedactionRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    RedactionRule(
        "nric_fin",
        re.compile(r"\b[STFGMstfgm]\d{7}[A-Za-z]\b"),
        "[NRIC_REDACTED]",
    ),
    RedactionRule(
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        "[CARD_REDACTED]",
    ),
    RedactionRule(
        # Singapore mobile/landline (8 digits starting 6/8/9), with optional +65.
        "phone",
        re.compile(r"(?:\+?65[\s-]?)?\b[689]\d{3}[\s-]?\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
    RedactionRule(
        "ip_address",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[IP_REDACTED]",
    ),
)

# Confidence annotators, keyed by rule label.
_CHECKS = {
    "credit_card": lambda m: luhn_valid(m),
    "nric_fin": nric_checksum_valid,
}


class PatternRedactor:
    """Regex redaction with checksum-based confidence annotation. No heavy dependencies."""

    name = "pattern-redactor"

    def __init__(self, rules: tuple[RedactionRule, ...] = DEFAULT_RULES) -> None:
        self._rules = rules

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        report = RedactionReport()
        result = text
        # Rule order matters: emails are matched before phone numbers so that digits
        # inside an address are not partially rewritten first.
        for rule in self._rules:
            check = _CHECKS.get(rule.label)

            # Both `rule` and `check` are bound as defaults: without that, the closure
            # would capture the loop variables by reference and every rule would use
            # the last iteration's checksum.
            def substitute(
                match: re.Match[str],
                rule: RedactionRule = rule,
                check: object = check,
            ) -> str:
                report.counts[rule.label] = report.counts.get(rule.label, 0) + 1
                if check is not None and not check(match.group(0)):  # type: ignore[operator]
                    report.low_confidence[rule.label] = report.low_confidence.get(rule.label, 0) + 1
                return rule.placeholder

            result = rule.pattern.sub(substitute, result)
        return result, report


class NullRedactor:
    """Explicit no-op, for corpora already cleared for storage. Never the default."""

    name = "null-redactor"

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        return text, RedactionReport()
