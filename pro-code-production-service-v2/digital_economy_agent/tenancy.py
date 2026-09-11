"""
What counts as a tenant identifier.

Shared by the HTTP path and the event path so the two cannot drift. A tenant id
becomes a Postgres session setting that a row-level-security policy compares against
and half of a unique key, so the shape is deliberately narrow.

This is not an injection defence -- every query that uses it is parameterised. It is
here so a malformed identifier fails loudly instead of resolving to a tenant that
exists but is not the caller's, or to a value that matches no rows and is
indistinguishable from an empty corpus.
"""

from __future__ import annotations

import re
from typing import TypeGuard

TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def is_valid_tenant_id(value: object) -> TypeGuard[str]:
    """
    True when ``value`` is a usable tenant identifier.

    A TypeGuard rather than a plain bool so callers narrow to ``str`` and the type
    checker enforces that an unvalidated claim -- which may be any JSON value -- is
    never passed on as a tenant.
    """
    return isinstance(value, str) and bool(TENANT_PATTERN.match(value))
