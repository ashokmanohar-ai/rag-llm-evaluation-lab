from __future__ import annotations

from pathlib import Path

from rag_eval.models import CaseResult, EvaluationReport
from rag_eval.reporting import write_html, write_json, write_junit


def test_all_report_formats_are_generated(tmp_path: Path) -> None:
    report = EvaluationReport(
        evaluation_id="test",
        dataset="qa.jsonl",
        configuration={"provider": "mock"},
        metrics={"pass_rate": 0.5},
        cases=[
            CaseResult(case_id="PASS", passed=True, metrics={}),
            CaseResult(case_id="FAIL", passed=False, metrics={}, reason_codes=["HALLUCINATION"]),
        ],
        quality_gate="FAIL",
        gate_failures=["hallucination_rate breached"],
    )
    json_path = write_json(report, tmp_path / "evaluation.json")
    html_path = write_html(report, tmp_path / "evaluation.html")
    junit_path = write_junit(report, tmp_path / "junit.xml")
    assert '"evaluation_id":"test"'.replace(":", ": ") in json_path.read_text()
    assert "RAG &amp; LLM Evaluation Report" in html_path.read_text()
    assert 'failures="1"' in junit_path.read_text()
