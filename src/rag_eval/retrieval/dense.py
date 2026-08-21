"""Cosine dense retrieval, optionally backed by FAISS."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rag_eval.embeddings.protocol import EmbeddingProvider
from rag_eval.ingestion.metadata import eligible_chunk
from rag_eval.models import Chunk, SearchResult


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingProvider, use_faiss: bool = True):
        self.chunks = chunks
        self.embeddings = embeddings
        self.matrix = embeddings.embed([chunk.content for chunk in chunks])
        self._index: Any | None = None
        if use_faiss:
            try:
                import faiss

                self._index = faiss.IndexFlatIP(self.matrix.shape[1])
                self._index.add(self.matrix.astype(np.float32))
            except ImportError:
                self._index = None

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        exclude_statuses: set[str] | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embeddings.embed([query])[0]
        scores: NDArray[np.float64] = self.matrix @ query_vector
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if eligible_chunk(
                self.chunks[index], filters=filters, exclude_statuses=exclude_statuses
            )
        ]
        candidates.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                retriever="dense",
                component_scores={"dense": score},
            )
            for rank, (index, score) in enumerate(candidates[:top_k], start=1)
        ]
