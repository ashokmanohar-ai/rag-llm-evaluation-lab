from __future__ import annotations

from rag_eval.evaluation.generation import (
    completeness,
    groundedness,
    hallucination_details,
    required_fact_coverage,
)
from rag_eval.generation.mock import DeterministicMockProvider
from rag_eval.models import Chunk, ContextBundle


def test_mock_generation_is_deterministic_and_cited() -> None:
    chunk = Chunk(
        chunk_id="HR-LEAVE-C001",
        document_id="HR-LEAVE",
        document_title="Leave",
        section="Leave",
        content="Employees receive 25 days of paid annual leave.",
        source_path="leave.md",
    )
    context = ContextBundle(
        text="[SOURCE: HR-LEAVE-C001]\n" + chunk.content,
        included_chunks=[chunk],
        dropped_chunks=[],
        estimated_tokens=8,
    )
    provider = DeterministicMockProvider()
    first = provider.generate("How many annual leave days?", context)
    second = provider.generate("How many annual leave days?", context)
    assert first == second
    assert "25 days" in first.answer
    assert first.citations == ["HR-LEAVE-C001"]


def test_mock_refuses_when_evidence_has_no_overlap() -> None:
    context = ContextBundle(text="", included_chunks=[], dropped_chunks=[], estimated_tokens=0)
    response = DeterministicMockProvider().generate("quantum teleportation zqxj", context)
    assert "insufficient" in response.answer.lower()
    assert not response.citations


def test_required_fact_coverage_reports_missing_facts() -> None:
    score, missing = required_fact_coverage("Employees get 25 days.", ["25 days", "five carryover"])
    assert score == 0.5
    assert missing == ["five carryover"]
    assert completeness("anything", []) == 1


def test_groundedness_flags_unsupported_claim() -> None:
    score, unsupported = groundedness(
        "Employees receive 30 days of leave.",
        "Employees receive 25 days of annual leave.",
        threshold=0.9,
    )
    assert score == 0
    assert unsupported


def test_hallucination_checks_forbidden_claims() -> None:
    result = hallucination_details(
        "Email security@acme.example.",
        "Use the hotline form.",
        ["security@acme.example"],
    )
    assert result["hallucination_detected"] is True
    assert result["forbidden_claims"] == ["security@acme.example"]
