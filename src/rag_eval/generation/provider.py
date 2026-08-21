"""Generation provider protocol and safe factory."""

from __future__ import annotations

from typing import Any, Protocol

from rag_eval.models import ContextBundle, RAGAnswer


class GenerationProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, question: str, context: ContextBundle) -> RAGAnswer: ...


def build_generation_provider(config: dict[str, Any]) -> GenerationProvider:
    provider = str(config.get("provider", "mock"))
    if provider == "mock":
        from rag_eval.generation.mock import DeterministicMockProvider

        return DeterministicMockProvider()
    if provider in {"azure_openai", "openai"}:
        from rag_eval.generation.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider.from_environment(provider)
    if provider == "anthropic":
        from rag_eval.generation.anthropic_provider import AnthropicProvider

        return AnthropicProvider.from_environment()
    raise ValueError(f"Unsupported LLM provider: {provider}")
