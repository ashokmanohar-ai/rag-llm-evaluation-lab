"""Deterministic extractive provider for CI and framework validation."""

from __future__ import annotations

import re

from rag_eval.models import ContextBundle, RAGAnswer, TokenUsage
from rag_eval.retrieval.bm25 import tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "do",
    "does",
    "employee",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
}


class DeterministicMockProvider:
    """Extracts evidence sentences. It does not represent real-model quality."""

    provider_name = "mock"
    model_name = "deterministic-extractive-v1"

    def generate(self, question: str, context: ContextBundle) -> RAGAnswer:
        query_terms = set(tokenize(question)) - STOPWORDS
        query_lower = question.lower()
        excluded_terms = (
            set(tokenize(query_lower.split("not", 1)[1])) - STOPWORDS
            if "not" in query_lower
            else set()
        )
        candidates: list[tuple[float, str, str]] = []
        best_evidence_overlap = 0.0
        for chunk_position, chunk in enumerate(context.included_chunks):
            clean = re.sub(r"^#{1,6}\s+", "", chunk.content, flags=re.MULTILINE)
            title_terms = set(tokenize(chunk.document_title))
            title_matches = query_terms & title_terms
            title_overlap = len(title_matches) / max(len(title_terms), 1)
            if len(title_matches) >= 2:
                best_evidence_overlap = max(best_evidence_overlap, title_overlap)
            for sentence_position, sentence in enumerate(SENTENCE.split(clean.replace("\n", " "))):
                sentence = sentence.strip()
                terms = set(tokenize(sentence))
                overlap = len(query_terms & terms) / max(len(query_terms), 1)
                if overlap:
                    best_evidence_overlap = max(best_evidence_overlap, overlap)
                    rank_bonus = max(0.0, 0.20 - chunk_position * 0.04)
                    position_bonus = max(0.0, 0.03 - sentence_position * 0.005)
                    intent_bonus = self._intent_bonus(query_lower, sentence.lower())
                    intent_bonus -= 0.20 * len(excluded_terms & terms)
                    score = (
                        overlap + rank_bonus + title_overlap * 0.20 + position_bonus + intent_bonus
                    )
                    candidates.append((score, sentence, chunk.chunk_id))
        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        if not candidates or best_evidence_overlap < 0.30 or len(query_terms) <= 1:
            answer = "The available knowledge is insufficient to answer this question."
            citations: list[str] = []
        else:
            selected: list[tuple[str, str]] = []
            seen: set[str] = set()
            for _, sentence, citation in candidates:
                key = sentence.lower()
                if key not in seen:
                    selected.append((sentence, citation))
                    seen.add(key)
                if len(selected) == 1:
                    break
            answer = " ".join(f"{sentence} [{citation}]" for sentence, citation in selected)
            citations = list(dict.fromkeys(citation for _, citation in selected))
        context_tokens = len(tokenize(context.text))
        output_tokens = len(tokenize(answer))
        return RAGAnswer(
            answer=answer,
            citations=citations,
            confidence=None,
            token_usage=TokenUsage(
                prompt_tokens=None,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                total_tokens=context_tokens + output_tokens,
            ),
        )

    @staticmethod
    def _intent_bonus(question: str, sentence: str) -> float:
        bonus = 0.0
        if question.startswith("where") and any(
            value in sentence for value in (" from ", " through ", " at ")
        ):
            bonus += 0.20
        if "qualif" in question and " means " in sentence:
            bonus += 0.25
        if "receipt" in question and "receipt" in sentence:
            bonus += 0.15
        if ("how many" in question or "how long" in question) and re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|thirty)\b", sentence
        ):
            bonus += 0.10
        return bonus
