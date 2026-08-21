"""Deterministic generation evaluators for controlled datasets."""

from __future__ import annotations

import re

from rag_eval.retrieval.bm25 import tokenize

CITATION = re.compile(r"\[[A-Z0-9][A-Z0-9-]+-C\d{3}\]")


def _normalise(value: str) -> str:
    return " ".join(tokenize(value))


def required_fact_coverage(answer: str, required_facts: list[str]) -> tuple[float, list[str]]:
    if not required_facts:
        return 1.0, []
    normalised = _normalise(answer)
    missing = [fact for fact in required_facts if _normalise(fact) not in normalised]
    return (len(required_facts) - len(missing)) / len(required_facts), missing


def completeness(answer: str, required_facts: list[str]) -> float:
    return required_fact_coverage(answer, required_facts)[0]


def split_claims(answer: str) -> list[str]:
    clean = CITATION.sub("", answer)
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", clean) if item.strip()]


def groundedness(answer: str, context: str, threshold: float = 0.65) -> tuple[float, list[str]]:
    claims = split_claims(answer)
    if not claims:
        return 1.0, []
    context_tokens = set(tokenize(context))
    unsupported: list[str] = []
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        overlap = len(claim_tokens & context_tokens) / max(len(claim_tokens), 1)
        claim_numbers = set(re.findall(r"\b\d[\d,.:]*\b", claim))
        context_numbers = set(re.findall(r"\b\d[\d,.:]*\b", context))
        numeric_mismatch = not claim_numbers.issubset(context_numbers)
        if overlap < threshold or numeric_mismatch:
            unsupported.append(claim)
    return (len(claims) - len(unsupported)) / len(claims), unsupported


def hallucination_details(
    answer: str,
    context: str,
    forbidden_claims: list[str] | None = None,
    threshold: float = 0.65,
) -> dict[str, object]:
    refusal = "insufficient" in answer.lower() or "cannot be determined" in answer.lower()
    if refusal and not forbidden_claims:
        return {
            "hallucination_detected": False,
            "groundedness": 1.0,
            "unsupported_claims": [],
            "forbidden_claims": [],
        }
    score, unsupported = groundedness(answer, context, threshold)
    normalised = _normalise(answer)
    forbidden = [claim for claim in (forbidden_claims or []) if _normalise(claim) in normalised]
    return {
        "hallucination_detected": bool(unsupported or forbidden),
        "groundedness": score,
        "unsupported_claims": unsupported,
        "forbidden_claims": forbidden,
    }
