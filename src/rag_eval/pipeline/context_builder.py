"""Traceable context assembly with deduplication and explicit budget drops."""

from __future__ import annotations

from rag_eval.models import ContextBundle, SearchResult
from rag_eval.retrieval.bm25 import tokenize


class ContextBuilder:
    def __init__(self, max_tokens: int = 6000, near_duplicate_threshold: float = 0.92):
        self.max_tokens = max_tokens
        self.near_duplicate_threshold = near_duplicate_threshold

    def build(self, results: list[SearchResult]) -> ContextBundle:
        included = []
        dropped: list[str] = []
        token_total = 0
        redundancy = 0
        token_sets: list[set[str]] = []
        for result in results:
            tokens = tokenize(result.chunk.content)
            token_set = set(tokens)
            duplicate = any(
                len(token_set & previous) / max(len(token_set | previous), 1)
                >= self.near_duplicate_threshold
                for previous in token_sets
            )
            if duplicate:
                redundancy += 1
                dropped.append(result.chunk.chunk_id)
                continue
            if token_total + len(tokens) > self.max_tokens:
                dropped.append(result.chunk.chunk_id)
                continue
            included.append(result.chunk)
            token_sets.append(token_set)
            token_total += len(tokens)
        text = "\n\n".join(f"[SOURCE: {chunk.chunk_id}]\n{chunk.content}" for chunk in included)
        return ContextBundle(
            text=text,
            included_chunks=included,
            dropped_chunks=dropped,
            estimated_tokens=token_total,
            redundancy_count=redundancy,
        )
