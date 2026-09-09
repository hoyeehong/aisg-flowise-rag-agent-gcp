"""Ingestion: parsing, PII redaction, chunking, embedding and idempotent upsert."""

from .pipeline import IngestionResult, ingest_pages, ingest_pdf, read_pdf_pages
from .redaction import (
    DEFAULT_RULES,
    NullRedactor,
    PatternRedactor,
    RedactionReport,
    RedactionRule,
    Redactor,
    luhn_valid,
    nric_checksum_valid,
)

__all__ = [
    "DEFAULT_RULES",
    "IngestionResult",
    "NullRedactor",
    "PatternRedactor",
    "RedactionReport",
    "RedactionRule",
    "Redactor",
    "ingest_pages",
    "ingest_pdf",
    "luhn_valid",
    "nric_checksum_valid",
    "read_pdf_pages",
]
