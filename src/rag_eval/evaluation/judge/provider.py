from typing import Protocol

from rag_eval.models import JudgeResult


class JudgeProvider(Protocol):
    def judge(self, question: str, answer: str, reference: str) -> JudgeResult: ...
