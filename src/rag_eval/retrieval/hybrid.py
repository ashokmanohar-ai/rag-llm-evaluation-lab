"""Configurable dense, sparse, or fused retrieval."""

from __future__ import annotations

from typing import Any

from rag_eval.embeddings.protocol import EmbeddingProvider
from rag_eval.models import Chunk, SearchResult
from rag_eval.retrieval.bm25 import BM25Retriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider, config: dict[str, Any]):
        self.config = config
        self.dense = DenseRetriever(chunks, embeddings)
        self.sparse = BM25Retriever(chunks)

    def search(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        mode = str(self.config.get("mode", "hybrid"))
        final_k = int(top_k or self.config.get("final_top_k", 5))
        excluded = set(self.config.get("exclude_statuses", []))
        if mode == "dense":
            return self.dense.search(query, final_k, filters, excluded)
        if mode == "sparse":
            return self.sparse.search(query, final_k, filters, excluded)
        dense = self.dense.search(query, int(self.config.get("dense_top_k", 20)), filters, excluded)
        sparse = self.sparse.search(
            query, int(self.config.get("sparse_top_k", 20)), filters, excluded
        )
        return reciprocal_rank_fusion(
            [dense, sparse],
            constant=int(self.config.get("rrf_constant", 60)),
            top_k=max(final_k, int(self.config.get("final_top_k", final_k))),
        )[:final_k]
