from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_eval.regression import QualityGate, compare_metrics, load_baseline, save_baseline


def test_baseline_round_trip_and_comparison(tmp_path: Path) -> None:
    target = tmp_path / "baseline.json"
    save_baseline(target, {"recall_at_5": 0.9}, dataset="qa.jsonl", configuration={})
    baseline = load_baseline(target)
    comparison = compare_metrics(baseline, {"recall_at_5": 0.8}, expected_dataset="qa.jsonl")
    assert comparison["recall_at_5"]["delta"] == pytest.approx(-0.1)


def test_comparison_rejects_different_dataset(tmp_path: Path) -> None:
    target = tmp_path / "baseline.json"
    save_baseline(target, {"recall_at_5": 0.9}, dataset="qa.jsonl", configuration={})
    with pytest.raises(ValueError, match="same dataset"):
        compare_metrics(load_baseline(target), {}, expected_dataset="other.jsonl")


def test_quality_gate_blocks_hallucination_breach(gate_config: dict[str, Any]) -> None:
    status, failures = QualityGate(gate_config).evaluate(
        {
            "recall_at_5": 1,
            "precision_at_5": 1,
            "completeness": 1,
            "groundedness": 1,
            "hallucination_rate": 0.5,
            "citation_correctness": 1,
            "citation_completeness": 1,
        }
    )
    assert status == "FAIL"
    assert any("hallucination_rate" in failure for failure in failures)


def test_latency_breach_is_conditional(gate_config: dict[str, Any]) -> None:
    status, _ = QualityGate(gate_config).evaluate({"latency_p95_ms": 9999})
    assert status == "CONDITIONAL_PASS"
