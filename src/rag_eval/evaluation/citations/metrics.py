"""Deterministic citation existence, correctness, and completeness."""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.evaluation.generation.deterministic import split_claims
from rag_eval.pipeline.citation_mapper import extract_citations
from rag_eval.retrieval.bm25 import tokenize


@dataclass(frozen=True)
class CitationEvaluation:
    existence: float
    correctness: float
    completeness: float
    invalid: list[str]
    attribution: dict[str, str | None]


def evaluate_citations(
    answer: str, available_sources: dict[str, str], expected_sources: set[str] | None = None
) -> CitationEvaluation:
    citations = extract_citations(answer)
    invalid = [citation for citation in citations if citation not in available_sources]
    existence = 1.0 if not invalid else (len(citations) - len(invalid)) / max(len(citations), 1)
    attribution = {citation: available_sources.get(citation) for citation in citations}
    if expected_sources:
        correct = [
            citation
            for citation in citations
            if any(
                citation == source or citation.startswith(f"{source}-C")
                for source in expected_sources
            )
        ]
        correctness = len(correct) / max(len(citations), 1)
    else:
        correctness = existence
    claims = split_claims(answer)
    cited_claims = 0
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        if any(
            citation in answer
            and len(claim_tokens & set(tokenize(source))) / max(len(claim_tokens), 1) >= 0.5
            for citation, source in attribution.items()
            if source
        ):
            cited_claims += 1
    completeness = cited_claims / len(claims) if claims else 1.0
    return CitationEvaluation(existence, correctness, completeness, invalid, attribution)
