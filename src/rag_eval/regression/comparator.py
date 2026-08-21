"""Compare matching metrics without silently mixing datasets."""

from __future__ import annotations

from typing import Any


def compare_metrics(
    baseline: dict[str, Any], current: dict[str, Any], *, expected_dataset: str
) -> dict[str, dict[str, float]]:
    if baseline.get("dataset") != expected_dataset:
        raise ValueError("Baseline and current evaluation must use the same dataset")
    output: dict[str, dict[str, float]] = {}
    for key, old_value in baseline.get("metrics", {}).items():
        new_value = current.get(key)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            output[key] = {
                "baseline": float(old_value),
                "current": float(new_value),
                "delta": float(new_value) - float(old_value),
            }
    return output
