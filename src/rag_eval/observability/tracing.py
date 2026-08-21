"""Optional OpenTelemetry facade that keeps core execution dependency-free."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    try:
        from opentelemetry import trace

        with trace.get_tracer("rag_eval").start_as_current_span(name, attributes=attributes):
            yield
    except ImportError:
        yield


def trace_attributes(config: dict[str, Any], query: str) -> dict[str, Any]:
    return {
        "rag.query.length": len(query),
        "rag.retrieval.mode": config.get("retrieval", {}).get("mode", "unknown"),
        "rag.retrieval.top_k": config.get("retrieval", {}).get("final_top_k", 5),
        "rag.embedding.model": config.get("embeddings", {}).get("model", "unknown"),
        "rag.reranker.enabled": config.get("reranking", {}).get("enabled", False),
        "gen_ai.provider": config.get("generation", {}).get("provider", "unknown"),
        "gen_ai.prompt.version": config.get("generation", {}).get("prompt_version", "unknown"),
    }
