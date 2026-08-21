"""Small transparent BM25 implementation for deterministic offline evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from rag_eval.ingestion.metadata import eligible_chunk
from rag_eval.models import Chunk, SearchResult

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(value: str) -> list[str]:
    tokens = TOKEN.findall(value.lower())
    normalised: list[str] = []
    for token in tokens:
        if len(token) == 1:
            continue
        if token.startswith("certif"):
            normalised.append("certif")
            continue
        if len(token) > 5 and token.endswith("ness"):
            token = token[:-4]
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalised.append(token)
    return normalised


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.content) for chunk in chunks]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        document_frequency: Counter[str] = Counter()
        for values in self.tokens:
            document_frequency.update(set(values))
        size = len(chunks)
        self.idf = {
            term: math.log(1 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score(self, query_tokens: list[str], index: int) -> float:
        length = len(self.tokens[index])
        score = 0.0
        for term in query_tokens:
            frequency = self.counts[index].get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / max(self.avg_length, 1)
            )
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        exclude_statuses: set[str] | None = None,
    ) -> list[SearchResult]:
        tokens = tokenize(query)
        candidates = [
            (index, self._score(tokens, index))
            for index, chunk in enumerate(self.chunks)
            if eligible_chunk(chunk, filters=filters, exclude_statuses=exclude_statuses)
        ]
        candidates = [item for item in candidates if item[1] > 0]
        candidates.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                retriever="bm25",
                component_scores={"bm25": score},
            )
            for rank, (index, score) in enumerate(candidates[:top_k], start=1)
        ]
