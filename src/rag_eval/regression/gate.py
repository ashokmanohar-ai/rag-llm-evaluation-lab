"""Declarative quality gates for absolute thresholds and blocking severity."""

from __future__ import annotations

from typing import Any, Literal

METRIC_NAMES = {
    "retrieval.recall_at_5": "recall_at_5",
    "retrieval.precision_at_5": "precision_at_5",
    "generation.completeness": "completeness",
    "generation.groundedness": "groundedness",
    "generation.hallucination_rate": "hallucination_rate",
    "citations.correctness": "citation_correctness",
    "citations.completeness": "citation_completeness",
    "performance.p95_latency_ms": "latency_p95_ms",
}


class QualityGate:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate(
        self, metrics: dict[str, float | int | bool | None]
    ) -> tuple[Literal["PASS", "FAIL", "CONDITIONAL_PASS"], list[str]]:
        blocking: list[str] = []
        warnings: list[str] = []
        for path, metric_name in METRIC_NAMES.items():
            section, rule_name = path.split(".")
            rule = self.config.get(section, {}).get(rule_name, {})
            value = metrics.get(metric_name)
            if value is None or not rule:
                continue
            failed = False
            if "minimum" in rule and float(value) < float(rule["minimum"]):
                failed = True
            if "maximum" in rule and float(value) > float(rule["maximum"]):
                failed = True
            if failed:
                message = f"{metric_name}={float(value):.4f} breached {rule}"
                (blocking if rule.get("blocking", True) else warnings).append(message)
        if blocking:
            return "FAIL", blocking + warnings
        if warnings:
            return "CONDITIONAL_PASS", warnings
        return "PASS", []
