"""OpenAI-compatible and Azure OpenAI provider adapters."""

from __future__ import annotations

import json
import os
from typing import Any

from rag_eval.generation.prompt_builder import build_prompt
from rag_eval.models import ContextBundle, RAGAnswer, TokenUsage


class OpenAICompatibleProvider:
    def __init__(self, client: Any, model_name: str, provider_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.provider_name = provider_name

    @classmethod
    def from_environment(cls, provider: str) -> OpenAICompatibleProvider:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the azure extra: pip install -e '.[azure]'") from exc
        if provider == "azure_openai":
            endpoint = _required("AZURE_OPENAI_ENDPOINT")
            deployment = _required("AZURE_OPENAI_DEPLOYMENT")
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=_required("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
            return cls(client, deployment, provider)
        model = _required("LLM_MODEL")
        client = OpenAI(api_key=_required("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        return cls(client, model, provider)

    def generate(self, question: str, context: ContextBundle) -> RAGAnswer:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return JSON with answer and citations fields."},
                {"role": "user", "content": build_prompt(question, context)},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        usage = response.usage
        return RAGAnswer(
            answer=str(payload["answer"]),
            citations=[str(value) for value in payload.get("citations", [])],
            token_usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                context_tokens=None,
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value
