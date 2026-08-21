from rag_eval.evaluation.citations import evaluate_citations
from rag_eval.pipeline import RAGPipeline
from rag_eval.pipeline.citation_mapper import extract_citations, invalid_citations


def test_citation_existence_and_correctness() -> None:
    sources = {"HR-LEAVE-C001": "Employees receive 25 days of annual leave."}
    result = evaluate_citations(
        "Employees receive 25 days of annual leave. [HR-LEAVE-C001]",
        sources,
        {"HR-LEAVE"},
    )
    assert result.existence == 1
    assert result.correctness == 1
    assert result.completeness == 1


def test_invented_citation_fails_existence(pipeline: RAGPipeline) -> None:
    citations = extract_citations("Invented fact. [HR-999-C001]")
    assert invalid_citations(citations, pipeline.chunks) == ["HR-999-C001"]


def test_wrong_source_fails_correctness() -> None:
    sources = {
        "SEC-INCIDENT-C001": "Incidents are reported immediately.",
        "HR-LEAVE-C001": "Employees receive 25 days of annual leave.",
    }
    result = evaluate_citations(
        "Employees receive 25 days. [SEC-INCIDENT-C001]", sources, {"HR-LEAVE"}
    )
    assert result.correctness == 0


def test_missing_citation_fails_completeness() -> None:
    result = evaluate_citations(
        "Employees receive 25 days.",
        {"HR-LEAVE-C001": "Employees receive 25 days."},
        {"HR-LEAVE"},
    )
    assert result.completeness == 0
