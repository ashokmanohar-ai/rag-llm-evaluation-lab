"""End-to-end offline-first RAG pipeline."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from rag_eval.embeddings import build_embedding_provider
from rag_eval.generation import build_generation_provider
from rag_eval.ingestion import Chunker, KnowledgeLoader
from rag_eval.models import PipelineTimings, QueryResponse, SearchResult
from rag_eval.pipeline.context_builder import ContextBuilder
from rag_eval.reranking import LexicalReranker, OptionalCrossEncoderReranker
from rag_eval.retrieval import HybridRetriever


class RAGPipeline:
    def __init__(self, knowledge_root: str | Path, config: dict[str, Any]):
        self.config = config
        loaded = KnowledgeLoader(knowledge_root).load_all()
        chunking = config.get("chunking", {})
        self.chunks = Chunker(
            strategy=str(chunking.get("strategy", "recursive")),
            chunk_size=int(chunking.get("chunk_size", 600)),
            overlap=int(chunking.get("chunk_overlap", 100)),
        ).chunk_documents(loaded)
        start = time.perf_counter()
        embeddings = build_embedding_provider(config.get("embeddings", {}))
        self.retriever = HybridRetriever(self.chunks, embeddings, config.get("retrieval", {}))
        self.embedding_build_ms = (time.perf_counter() - start) * 1000
        reranking = config.get("reranking", {})
        self.reranking_config = reranking
        self.reranker = (
            OptionalCrossEncoderReranker(str(reranking.get("model")))
            if reranking.get("backend") == "cross_encoder"
            else LexicalReranker()
        )
        context = config.get("context", {})
        self.context_builder = ContextBuilder(
            max_tokens=int(context.get("max_tokens", 6000)),
            near_duplicate_threshold=float(context.get("near_duplicate_threshold", 0.92)),
        )
        self.provider = build_generation_provider(config.get("generation", {}))

    def retrieve(
        self, question: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> list[SearchResult]:
        candidate_k = int(self.reranking_config.get("candidate_k", 20))
        requested = top_k or int(self.config.get("retrieval", {}).get("final_top_k", 5))
        results = self.retriever.search(
            question,
            candidate_k if self.reranking_config.get("enabled", True) else requested,
            filters,
        )
        if self.reranking_config.get("enabled", True):
            return self.reranker.rerank(question, results, requested)
        return results[:requested]

    def query(
        self, question: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> QueryResponse:
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        results = self.retrieve(question, filters, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        context_start = time.perf_counter()
        context = self.context_builder.build(results)
        context_ms = (time.perf_counter() - context_start) * 1000
        generation_start = time.perf_counter()
        answer = self.provider.generate(question, context)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        timings = PipelineTimings(
            embedding_ms=0,
            retrieval_ms=retrieval_ms,
            reranking_ms=0,
            context_ms=context_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )
        return QueryResponse(
            answer=answer.answer,
            citations=answer.citations,
            retrieved_chunks=len(context.included_chunks),
            dropped_chunks=context.dropped_chunks,
            latency_ms=total_ms,
            timings=timings,
            token_usage=answer.token_usage,
            trace_id=uuid.uuid4().hex,
        )
