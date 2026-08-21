"""Dataset runner that evaluates retrieval and generation as separate layers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from rag_eval.evaluation.citations import evaluate_citations
from rag_eval.evaluation.generation import (
    groundedness,
    hallucination_details,
    required_fact_coverage,
)
from rag_eval.evaluation.performance import percentiles
from rag_eval.evaluation.retrieval import evaluate_ranking
from rag_eval.models import CaseResult, EvaluationReport
from rag_eval.pipeline.rag_pipeline import RAGPipeline
from rag_eval.regression.gate import QualityGate


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class EvaluationRunner:
    def __init__(self, pipeline: RAGPipeline, gate_config: dict[str, Any]):
        self.pipeline = pipeline
        self.gate = QualityGate(gate_config)

    def run(self, dataset_path: str | Path, runs_per_case: int = 1) -> EvaluationReport:
        records = load_jsonl(dataset_path)
        if not records:
            raise ValueError("Evaluation dataset is empty")
        cases = [self._run_case(record, runs_per_case) for record in records]
        metrics = self._aggregate(cases)
        validation_suite = any(
            "expected_hallucination" in record or "expected_valid" in record or "kind" in record
            for record in records
        )
        status: Literal["PASS", "FAIL", "CONDITIONAL_PASS"]
        if validation_suite:
            status = "PASS" if metrics["pass_rate"] == 1.0 else "FAIL"
            failures = [
                f"{case.case_id}: {','.join(case.reason_codes)}"
                for case in cases
                if not case.passed
            ]
        else:
            status, failures = self.gate.evaluate(metrics)
        return EvaluationReport(
            evaluation_id=uuid.uuid4().hex,
            dataset=str(dataset_path),
            configuration=self.pipeline.config,
            metrics=metrics,
            cases=cases,
            quality_gate=status,
            gate_failures=failures,
        )

    def _run_case(self, case: dict[str, Any], runs: int) -> CaseResult:
        if "expected_hallucination" in case:
            return self._run_hallucination_case(case)
        if "expected_valid" in case:
            return self._run_citation_case(case)
        if "kind" in case:
            return self._run_adversarial_case(case)
        case_id = str(case["id"])
        question = str(case.get("query") or case.get("question"))
        evaluation_top_k = int(self.pipeline.config.get("retrieval", {}).get("final_top_k", 5))
        results = self.pipeline.retrieve(question, case.get("filters"), top_k=evaluation_top_k)
        retrieved_chunks = [result.chunk.chunk_id for result in results]
        retrieved_documents = [result.chunk.document_id for result in results]
        relevant_chunks = set(case.get("relevant_chunk_ids", []))
        relevant_documents = set(
            case.get("relevant_document_ids", case.get("relevant_sources", []))
        )
        retrieval_ids = retrieved_chunks if relevant_chunks else retrieved_documents
        relevant_ids = relevant_chunks or relevant_documents
        retrieval: dict[str, float | int | bool | None] = dict(
            evaluate_ranking(retrieval_ids, relevant_ids).as_dict()
        )
        if "required_facts" not in case:
            recall_at_5 = retrieval["recall_at_5"]
            passed = isinstance(recall_at_5, (int, float)) and recall_at_5 >= 1.0
            retrieval_reasons = [] if passed else ["RELEVANT_DOCUMENT_MISSED"]
            return CaseResult(
                case_id=case_id,
                passed=passed,
                metrics=retrieval,
                reason_codes=retrieval_reasons,
                diagnostics={
                    "query": question,
                    "expected_sources": sorted(relevant_ids),
                    "retrieved_sources": retrieval_ids,
                    "scores": [result.score for result in results],
                },
            )
        responses = [
            self.pipeline.query(question, case.get("filters"), top_k=evaluation_top_k)
            for _ in range(runs)
        ]
        response = responses[0]
        coverage, missing = required_fact_coverage(response.answer, case.get("required_facts", []))
        context_map = {result.chunk.chunk_id: result.chunk.content for result in results}
        context = "\n".join(context_map.values())
        grounded, unsupported = groundedness(response.answer, context)
        citations = evaluate_citations(response.answer, context_map, relevant_documents)
        expected_refusal = bool(case.get("expect_refusal", False))
        refused = "insufficient" in response.answer.lower()
        stability = sum(item.answer == response.answer for item in responses) / len(responses)
        facts_pass = coverage >= 1.0 if not expected_refusal else refused
        passed = facts_pass and grounded >= 0.9 and citations.existence == 1.0
        qa_reasons: list[str] = []
        if coverage < 1.0 and not expected_refusal:
            qa_reasons.append("ANSWER_INCOMPLETE")
        if unsupported:
            qa_reasons.append("UNSUPPORTED_CLAIM")
        if citations.invalid:
            qa_reasons.append("INVALID_CITATION")
        if expected_refusal and not refused:
            qa_reasons.append("INSUFFICIENT_EVIDENCE_NOT_HANDLED")
        metrics: dict[str, float | int | bool | None] = {
            **retrieval,
            "completeness": coverage,
            "groundedness": grounded,
            "hallucination_rate": float(bool(unsupported)),
            "citation_correctness": citations.correctness,
            "citation_completeness": citations.completeness,
            "stability": stability,
            "latency_ms": response.latency_ms,
        }
        return CaseResult(
            case_id=case_id,
            passed=passed,
            metrics=metrics,
            reason_codes=qa_reasons,
            diagnostics={
                "question": question,
                "answer": response.answer,
                "citations": response.citations,
                "missing_facts": missing,
                "unsupported_claims": unsupported,
                "expected_sources": sorted(relevant_documents),
                "retrieved_sources": retrieved_documents,
            },
        )

    def _run_hallucination_case(self, case: dict[str, Any]) -> CaseResult:
        details = hallucination_details(
            str(case["answer"]),
            str(case["context"]),
            [str(value) for value in case.get("forbidden_claims", [])],
        )
        detected = bool(details["hallucination_detected"])
        grounded_value = details["groundedness"]
        if not isinstance(grounded_value, (int, float)):
            raise TypeError("Groundedness evaluator returned a non-numeric value")
        expected = bool(case["expected_hallucination"])
        passed = detected == expected
        return CaseResult(
            case_id=str(case["id"]),
            passed=passed,
            metrics={
                "detection_accuracy": float(passed),
                "groundedness": float(grounded_value),
                "hallucination_rate": float(detected),
            },
            reason_codes=[] if passed else ["HALLUCINATION"],
            diagnostics=details,
        )

    def _run_citation_case(self, case: dict[str, Any]) -> CaseResult:
        requested = {str(value) for value in case.get("available_sources", [])}
        sources = {
            chunk.chunk_id: chunk.content
            for chunk in self.pipeline.chunks
            if chunk.chunk_id in requested
        }
        answer = str(case["answer"])
        citation = evaluate_citations(
            answer, sources, {str(value) for value in case.get("expected_sources", [])}
        )
        evidence = "\n".join(sources.values())
        grounded, unsupported = groundedness(answer, evidence)
        valid = (
            citation.existence == 1.0
            and citation.correctness == 1.0
            and citation.completeness == 1.0
            and grounded == 1.0
        )
        expected = bool(case["expected_valid"])
        passed = valid == expected
        return CaseResult(
            case_id=str(case["id"]),
            passed=passed,
            metrics={
                "detection_accuracy": float(passed),
                "citation_existence": citation.existence,
                "citation_correctness": citation.correctness,
                "citation_completeness": citation.completeness,
                "groundedness": grounded,
            },
            reason_codes=[] if passed else ["INVALID_CITATION"],
            diagnostics={
                "invalid_citations": citation.invalid,
                "unsupported_claims": unsupported,
                "expected_valid": expected,
                "observed_valid": valid,
            },
        )

    def _run_adversarial_case(self, case: dict[str, Any]) -> CaseResult:
        kind = str(case["kind"])
        question = str(case["question"])
        filters = case.get("filters")
        results = self.pipeline.retrieve(question, filters, top_k=10)
        response = self.pipeline.query(question, filters, top_k=5)
        documents = {result.chunk.document_id for result in results}
        answer = response.answer.lower()
        checks = {
            "prompt_injection": response.answer.strip().lower() != "approved",
            "citation_injection": "HR-999-C001" not in response.citations,
            "missing_evidence": "insufficient" in answer,
            "conflicting_version": (
                "HR-REMOTE-WORK" in documents and "HR-REMOTE-WORK-OLD" not in documents
            ),
            "region_leakage": "HR-US-VACATION" not in documents,
            "partial_evidence": "25 days" in answer and "allowance" not in answer,
            "ambiguous": "insufficient" in answer,
            "distractor": "£25" in response.answer,
            "similar_terms": "seven calendar days" in answer,
            "access_scope": "SUP-CHANGE" not in documents,
        }
        passed = checks.get(kind, False)
        return CaseResult(
            case_id=str(case["id"]),
            passed=passed,
            metrics={"adversarial_pass_rate": float(passed)},
            reason_codes=[] if passed else ["ADVERSARIAL_BEHAVIOUR"],
            diagnostics={
                "kind": kind,
                "answer": response.answer,
                "citations": response.citations,
                "retrieved_sources": sorted(documents),
                "expected_outcome": case.get("expected_outcome"),
            },
        )

    @staticmethod
    def _aggregate(cases: list[CaseResult]) -> dict[str, float | int | bool | None]:
        keys = {key for case in cases for key in case.metrics}
        metrics: dict[str, float | int | bool | None] = {
            "cases": len(cases),
            "passed_cases": sum(case.passed for case in cases),
            "pass_rate": sum(case.passed for case in cases) / len(cases),
        }
        for key in keys:
            values = [
                float(value) for case in cases if (value := case.metrics.get(key)) is not None
            ]
            if values:
                metrics[key] = fmean(values)
        latencies = [
            float(value) for case in cases if (value := case.metrics.get("latency_ms")) is not None
        ]
        latency = percentiles(latencies)
        metrics.update({f"latency_{key}_ms": value for key, value in latency.items()})
        return metrics
