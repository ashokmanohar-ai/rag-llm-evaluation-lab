from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_eval.evaluation import EvaluationRunner
from rag_eval.evaluation.performance import cost, percentiles
from rag_eval.pipeline import RAGPipeline


def test_retrieval_dataset_runs_offline(
    pipeline: RAGPipeline, gate_config: dict[str, Any], project_root: Path
) -> None:
    report = EvaluationRunner(pipeline, gate_config).run(
        project_root / "data/evaluation/retrieval.jsonl"
    )
    assert report.metrics["cases"] == 25
    recall = report.metrics["recall_at_5"]
    assert isinstance(recall, (int, float))
    assert recall >= 0.8
    assert report.quality_gate in {"PASS", "CONDITIONAL_PASS"}


def test_qa_dataset_has_case_diagnostics(
    pipeline: RAGPipeline, gate_config: dict[str, Any], project_root: Path
) -> None:
    report = EvaluationRunner(pipeline, gate_config).run(
        project_root / "data/evaluation/qa.jsonl", runs_per_case=2
    )
    assert report.metrics["cases"] == 25
    assert all("question" in case.diagnostics for case in report.cases)
    assert all(case.metrics["stability"] == 1 for case in report.cases)


def test_negative_and_adversarial_validation_suites_pass(
    pipeline: RAGPipeline, gate_config: dict[str, Any], project_root: Path
) -> None:
    runner = EvaluationRunner(pipeline, gate_config)
    for name in ("hallucination.jsonl", "citations.jsonl", "adversarial.jsonl"):
        report = runner.run(project_root / "data/evaluation" / name)
        assert report.quality_gate == "PASS", (name, report.gate_failures)
        assert report.metrics["pass_rate"] == 1.0


def test_latency_percentiles_and_cost_do_not_invent_data() -> None:
    values = percentiles([1, 2, 3, 4])
    assert values["p50"] == 2.5
    assert values["p95"] is not None
    assert cost(None, 10, 1.0, 2.0) is None
    assert cost(1_000_000, 500_000, 2.0, 4.0) == 4.0
