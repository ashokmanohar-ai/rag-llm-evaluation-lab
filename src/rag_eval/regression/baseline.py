"""Versioned, provenance-rich baseline persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save_baseline(
    path: str | Path,
    metrics: dict[str, Any],
    *,
    dataset: str,
    configuration: dict[str, Any],
) -> None:
    payload = {
        "version": "rag-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "configuration": configuration,
        "metrics": metrics,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Baseline must be a JSON object")
    return value
