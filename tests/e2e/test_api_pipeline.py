from __future__ import annotations

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from rag_eval.api import app as api_module
from rag_eval.pipeline import RAGPipeline


def test_pipeline_query_is_grounded_and_cited(pipeline: RAGPipeline) -> None:
    response = pipeline.query("How many paid annual leave days do UK employees receive?")
    assert "25 days" in response.answer
    assert response.citations
    assert all(citation.startswith("HR-ANNUAL-LEAVE") for citation in response.citations)
    assert response.latency_ms >= 0


def test_api_health_and_query(monkeypatch: MonkeyPatch, pipeline: RAGPipeline) -> None:
    monkeypatch.setattr(api_module, "_pipeline", pipeline)
    client = TestClient(api_module.app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post(
        "/api/v1/query", json={"question": "How long is a temporary password valid?"}
    )
    assert response.status_code == 200
    assert "30 minutes" in response.json()["answer"]


def test_rebuild_requires_configured_key(monkeypatch: MonkeyPatch, pipeline: RAGPipeline) -> None:
    monkeypatch.setattr(api_module, "_pipeline", pipeline)
    monkeypatch.delenv("INDEX_REBUILD_API_KEY", raising=False)
    response = TestClient(api_module.app).post("/api/v1/index/rebuild")
    assert response.status_code == 401


def test_evaluation_rejects_path_traversal(monkeypatch: MonkeyPatch, pipeline: RAGPipeline) -> None:
    monkeypatch.setattr(api_module, "_pipeline", pipeline)
    response = TestClient(api_module.app).post(
        "/api/v1/evaluate", json={"dataset": "../../etc/passwd"}
    )
    assert response.status_code == 400
