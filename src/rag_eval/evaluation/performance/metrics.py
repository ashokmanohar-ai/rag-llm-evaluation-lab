"""Latency percentile and provider-cost calculations."""

from __future__ import annotations

import numpy as np


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_per_million: float | None,
    output_per_million: float | None,
) -> float | None:
    values = (input_tokens, output_tokens, input_per_million, output_per_million)
    if any(value is None for value in values):
        return None
    assert input_tokens is not None
    assert output_tokens is not None
    assert input_per_million is not None
    assert output_per_million is not None
    return (
        input_tokens / 1_000_000 * input_per_million
        + output_tokens / 1_000_000 * output_per_million
    )
