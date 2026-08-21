from __future__ import annotations

from rag_eval.evaluation.retrieval.metrics import (
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.pipeline import RAGPipeline
from rag_eval.retrieval.fusion import reciprocal_rank_fusion


def test_known_query_retrieves_expected_source(pipeline: RAGPipeline) -> None:
    results = pipeline.retrieve("How many days of annual leave do UK employees receive?")
    assert results[0].chunk.document_id == "HR-ANNUAL-LEAVE"


def test_metadata_filter_prevents_region_leakage(pipeline: RAGPipeline) -> None:
    results = pipeline.retrieve(
        "employee vacation annual leave", filters={"region": "UK"}, top_k=10
    )
    assert results
    assert all(result.chunk.metadata["region"] == "UK" for result in results)
    assert "HR-US-VACATION" not in {result.chunk.document_id for result in results}


def test_superseded_documents_are_excluded(pipeline: RAGPipeline) -> None:
    results = pipeline.retrieve("current remote working days", top_k=20)
    ids = {result.chunk.document_id for result in results}
    assert "HR-REMOTE-WORK" in ids
    assert "HR-REMOTE-WORK-OLD" not in ids


def test_reciprocal_rank_fusion_merges_without_duplicates(pipeline: RAGPipeline) -> None:
    dense = pipeline.retriever.dense.search("password reset", 5)
    sparse = pipeline.retriever.sparse.search("password reset", 5)
    fused = reciprocal_rank_fusion([dense, sparse], constant=60, top_k=5)
    assert len({result.chunk.chunk_id for result in fused}) == len(fused)
    assert fused[0].rank == 1
    assert "dense" in fused[0].component_scores or "bm25" in fused[0].component_scores


def test_reciprocal_rank_fusion_validates_constant() -> None:
    try:
        reciprocal_rank_fusion([], constant=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_retrieval_metric_formulas() -> None:
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "D"}
    assert precision_at_k(retrieved, relevant, 5) == 0.4
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 1) == 0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, {"B": 2, "D": 1}, 5) > 0


def test_retrieval_metrics_empty_relevance_is_well_defined() -> None:
    metrics = evaluate_ranking(["A"], set()).as_dict()
    assert metrics["recall_at_5"] == 1
    assert metrics["precision_at_5"] == 0
