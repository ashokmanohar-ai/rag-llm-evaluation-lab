from typing import Protocol

from rag_eval.models import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
