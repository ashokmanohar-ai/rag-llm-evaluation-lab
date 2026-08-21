"""Rerankers; lexical is deterministic, CrossEncoder requires the ml extra."""

from __future__ import annotations

from typing import Any

from rag_eval.models import SearchResult
from rag_eval.retrieval.bm25 import tokenize


class LexicalReranker:
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        rescored: list[tuple[SearchResult, float]] = []
        for result in results:
            content_terms = set(tokenize(result.chunk.content))
            title_terms = set(tokenize(result.chunk.document_title))
            content_overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            title_overlap = len(query_terms & title_terms) / max(len(title_terms), 1)
            rank_prior = 1 / max(result.rank, 1)
            score = content_overlap * 0.45 + title_overlap * 0.45 + rank_prior * 0.10
            rescored.append((result, score))
        rescored.sort(key=lambda item: (-item[1], item[0].chunk.chunk_id))
        return [
            result.model_copy(
                update={
                    "score": score,
                    "rank": rank,
                    "retriever": "lexical_reranker",
                    "component_scores": {**result.component_scores, "reranker": score},
                }
            )
            for rank, (result, score) in enumerate(rescored[:top_k], start=1)
        ]


class OptionalCrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the ml extra: pip install -e '.[ml]'") from exc
        self.model: Any = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        scores = self.model.predict([(query, result.chunk.content) for result in results])
        ranked = sorted(zip(results, scores, strict=True), key=lambda item: -float(item[1]))
        return [
            result.model_copy(
                update={
                    "score": float(score),
                    "rank": rank,
                    "retriever": "cross_encoder",
                    "component_scores": {**result.component_scores, "reranker": float(score)},
                }
            )
            for rank, (result, score) in enumerate(ranked[:top_k], start=1)
        ]
