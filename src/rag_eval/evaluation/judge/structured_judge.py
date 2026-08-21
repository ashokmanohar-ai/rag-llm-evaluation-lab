"""Deterministic mock judge used to test orchestration, never as real quality evidence."""

from rag_eval.models import JudgeResult
from rag_eval.retrieval.bm25 import tokenize


class DeterministicMockJudge:
    prompt_version = "mock-semantic-judge-v1"

    def judge(self, question: str, answer: str, reference: str) -> JudgeResult:
        del question
        expected = set(tokenize(reference))
        actual = set(tokenize(answer))
        score = len(expected & actual) / max(len(expected), 1)
        return JudgeResult(
            score=score,
            passed=score >= 0.6,
            rationale="Deterministic token overlap used only to validate judge plumbing in CI.",
            prompt_version=self.prompt_version,
        )
