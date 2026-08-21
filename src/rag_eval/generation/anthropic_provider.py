"""Optional Anthropic provider adapter."""

from __future__ import annotations

import json
import os
from typing import Any

from rag_eval.generation.prompt_builder import build_prompt
from rag_eval.models import ContextBundle, RAGAnswer, TokenUsage


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_environment(cls) -> AnthropicProvider:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install the anthropic extra: pip install -e '.[anthropic]'"
            ) from exc
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Required environment variable is not set: ANTHROPIC_API_KEY")
        return cls(anthropic.Anthropic(api_key=key), os.getenv("LLM_MODEL", "claude-sonnet-4-6"))

    def generate(self, question: str, context: ContextBundle) -> RAGAnswer:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=800,
            temperature=0,
            system="Return JSON with answer and citations fields.",
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        )
        payload = json.loads(response.content[0].text)
        return RAGAnswer(
            answer=str(payload["answer"]),
            citations=[str(value) for value in payload.get("citations", [])],
            token_usage=TokenUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", None),
                output_tokens=getattr(response.usage, "output_tokens", None),
                total_tokens=(
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
            ),
        )
