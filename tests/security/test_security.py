from __future__ import annotations

import logging

import pytest

from rag_eval.generation.mock import DeterministicMockProvider
from rag_eval.observability.logging import SecretRedactionFilter
from rag_eval.pipeline.context_builder import ContextBuilder
from rag_eval.pipeline.rag_pipeline import RAGPipeline


@pytest.mark.security
def test_prompt_injection_document_does_not_control_answer(pipeline: RAGPipeline) -> None:
    results = pipeline.retrieve(
        "What does the untrusted document say about internal citation instructions?",
        filters={"access_groups": "security-test"},
    )
    context = ContextBuilder().build(results)
    answer = DeterministicMockProvider().generate(
        "What is the security purpose of this untrusted document?", context
    )
    assert answer.answer.strip().lower() != "approved"
    assert "HR-999-C001" not in answer.citations


@pytest.mark.security
def test_access_group_filter_prevents_cross_scope_retrieval(pipeline: RAGPipeline) -> None:
    results = pipeline.retrieve(
        "production change implementation rollback", filters={"access_groups": "employees"}
    )
    assert "SUP-CHANGE" not in {result.chunk.document_id for result in results}


@pytest.mark.security
def test_secret_redaction_filter_masks_values() -> None:
    record = logging.LogRecord("test", logging.INFO, "", 0, "api_key=topsecret", (), None)
    SecretRedactionFilter().filter(record)
    assert "topsecret" not in str(record.msg)
    assert "***" in str(record.msg)
