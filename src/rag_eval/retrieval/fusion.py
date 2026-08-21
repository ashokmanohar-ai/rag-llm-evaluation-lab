"""Reciprocal Rank Fusion (RRF): score(d) = sum(1 / (k + rank))."""

from __future__ import annotations

from rag_eval.models import SearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]], *, constant: int = 60, top_k: int = 5
) -> list[SearchResult]:
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    by_id: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for results in result_sets:
        for result in results:
            chunk_id = result.chunk.chunk_id
            by_id[chunk_id] = result
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (constant + result.rank)
            components.setdefault(chunk_id, {}).update(result.component_scores)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        SearchResult(
            chunk=by_id[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            retriever="hybrid_rrf",
            component_scores=components[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
