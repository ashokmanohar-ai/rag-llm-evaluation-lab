"""Metadata filtering and current-version policy."""

from __future__ import annotations

from datetime import date
from typing import Any

from rag_eval.models import Chunk


def metadata_matches(chunk: Chunk, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = chunk.metadata.get(key)
        if isinstance(actual, list):
            expected_values = expected if isinstance(expected, list) else [expected]
            if not set(str(item) for item in expected_values).intersection(
                str(item) for item in actual
            ):
                return False
        elif actual != expected:
            return False
    return True


def eligible_chunk(
    chunk: Chunk,
    *,
    filters: dict[str, Any] | None = None,
    exclude_statuses: set[str] | None = None,
    as_of: date | None = None,
) -> bool:
    if chunk.metadata.get("status", "CURRENT") in (exclude_statuses or set()):
        return False
    effective = chunk.metadata.get("effective_date")
    if effective and as_of and date.fromisoformat(str(effective)) > as_of:
        return False
    return metadata_matches(chunk, filters or {})
