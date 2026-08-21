"""Transparent retrieval metrics for binary and graded relevance."""

from __future__ import annotations

import math
from dataclasses import dataclass


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(item in relevant for item in top) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], graded_relevance: dict[str, int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return float(
            sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(values, 1))
        )

    actual = dcg([graded_relevance.get(item, 0) for item in retrieved[:k]])
    ideal = dcg(sorted(graded_relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    hit_at_5: float
    reciprocal_rank: float
    ndcg_at_5: float

    def as_dict(self) -> dict[str, float]:
        return {
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "hit_at_5": self.hit_at_5,
            "mrr": self.reciprocal_rank,
            "ndcg_at_5": self.ndcg_at_5,
        }


def evaluate_ranking(
    retrieved: list[str], relevant: set[str], graded_relevance: dict[str, int] | None = None
) -> RetrievalMetrics:
    grades = graded_relevance or {item: 1 for item in relevant}
    return RetrievalMetrics(
        precision_at_1=precision_at_k(retrieved, relevant, 1),
        precision_at_3=precision_at_k(retrieved, relevant, 3),
        precision_at_5=precision_at_k(retrieved, relevant, 5),
        recall_at_1=recall_at_k(retrieved, relevant, 1),
        recall_at_3=recall_at_k(retrieved, relevant, 3),
        recall_at_5=recall_at_k(retrieved, relevant, 5),
        hit_at_5=hit_rate_at_k(retrieved, relevant, 5),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        ndcg_at_5=ndcg_at_k(retrieved, grades, 5),
    )
