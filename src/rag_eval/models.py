"""Shared, serialisable domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A normalized source document."""

    document_id: str
    title: str
    content: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A traceable section of a source document."""

    chunk_id: str
    document_id: str
    document_title: str
    section: str
    content: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str
    component_scores: dict[str, float] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    context_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class RAGAnswer(BaseModel):
    """Provider-independent grounded answer contract."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class ContextBundle(BaseModel):
    text: str
    included_chunks: list[Chunk]
    dropped_chunks: list[str]
    estimated_tokens: int
    redundancy_count: int = 0


class PipelineTimings(BaseModel):
    embedding_ms: float = 0
    retrieval_ms: float = 0
    reranking_ms: float = 0
    context_ms: float = 0
    generation_ms: float = 0
    total_ms: float = 0


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunks: int
    dropped_chunks: list[str]
    latency_ms: float
    timings: PipelineTimings
    token_usage: TokenUsage
    trace_id: str


class EvaluationRequest(BaseModel):
    dataset: str = "data/evaluation/qa.jsonl"
    runs_per_case: int = Field(default=1, ge=1, le=10)


class EvaluationStatus(BaseModel):
    id: str
    status: Literal["running", "completed", "failed"]
    report_path: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JudgeResult(BaseModel):
    score: float = Field(ge=0, le=1)
    passed: bool
    rationale: str
    prompt_version: str


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    passed: bool
    metrics: dict[str, float | int | bool | None]
    reason_codes: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    evaluation_id: str
    dataset: str
    configuration: dict[str, Any]
    metrics: dict[str, float | int | bool | None]
    cases: list[CaseResult]
    quality_gate: Literal["PASS", "FAIL", "CONDITIONAL_PASS"]
    gate_failures: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
