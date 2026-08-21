"""FastAPI endpoints for query, evaluation, report, and protected rebuild."""

from __future__ import annotations

import os
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse

from rag_eval.config import load_rag_config, load_yaml, project_path
from rag_eval.evaluation import EvaluationRunner
from rag_eval.models import EvaluationRequest, EvaluationStatus, QueryRequest, QueryResponse
from rag_eval.pipeline import RAGPipeline
from rag_eval.reporting import write_html, write_json, write_junit

app = FastAPI(
    title="RAG & LLM Evaluation Lab",
    version="1.0.0",
    description="Layered, offline-first RAG quality evaluation API.",
)
_lock = Lock()
_pipeline: RAGPipeline | None = None
_evaluations: dict[str, EvaluationStatus] = {}


def pipeline() -> RAGPipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = RAGPipeline(project_path("data/knowledge"), load_rag_config())
        return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return pipeline().query(request.question, request.filters, request.top_k)


@app.post("/api/v1/evaluate", response_model=EvaluationStatus)
def evaluate(request: EvaluationRequest) -> EvaluationStatus:
    requested = project_path(request.dataset).resolve()
    allowed_root = project_path("data/evaluation").resolve()
    if not requested.is_relative_to(allowed_root) or requested.suffix != ".jsonl":
        raise HTTPException(status_code=400, detail="Dataset must be a local evaluation JSONL file")
    evaluation_id = os.urandom(12).hex()
    state = EvaluationStatus(id=evaluation_id, status="running")
    _evaluations[evaluation_id] = state
    try:
        report = EvaluationRunner(pipeline(), load_yaml("config/quality-gates.yaml")).run(
            requested, request.runs_per_case
        )
        report.evaluation_id = evaluation_id
        target = project_path("reports") / evaluation_id
        write_json(report, target / "evaluation.json")
        write_html(report, target / "evaluation.html")
        write_junit(report, target / "junit.xml")
        state = EvaluationStatus(
            id=evaluation_id, status="completed", report_path=str(target / "evaluation.json")
        )
    except Exception as exc:
        state = EvaluationStatus(id=evaluation_id, status="failed", error=str(exc))
    _evaluations[evaluation_id] = state
    return state


@app.get("/api/v1/evaluations/{evaluation_id}", response_model=EvaluationStatus)
def evaluation_status(evaluation_id: str) -> EvaluationStatus:
    if evaluation_id not in _evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _evaluations[evaluation_id]


@app.get("/api/v1/reports/{evaluation_id}")
def report(evaluation_id: str) -> FileResponse:
    if not evaluation_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid evaluation ID")
    target = project_path("reports") / evaluation_id / "evaluation.html"
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(target)


@app.post("/api/v1/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
def rebuild_index(x_api_key: str | None = Header(default=None)) -> dict[str, int | str]:
    expected = os.getenv("INDEX_REBUILD_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="A valid index rebuild API key is required")
    global _pipeline
    with _lock:
        _pipeline = RAGPipeline(project_path("data/knowledge"), load_rag_config())
    return {"status": "rebuilt", "chunks": len(_pipeline.chunks)}
