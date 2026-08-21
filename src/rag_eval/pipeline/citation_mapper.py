"""Deterministic citation validation and attribution."""

from __future__ import annotations

import re

from rag_eval.models import Chunk

CITATION = re.compile(r"\[([A-Z0-9][A-Z0-9-]+-C\d{3})\]")


def extract_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(CITATION.findall(answer)))


def citation_attribution(citations: list[str], chunks: list[Chunk]) -> dict[str, str | None]:
    sources = {chunk.chunk_id: chunk.content for chunk in chunks}
    return {citation: sources.get(citation) for citation in citations}


def invalid_citations(citations: list[str], chunks: list[Chunk]) -> list[str]:
    known = {chunk.chunk_id for chunk in chunks}
    return [citation for citation in citations if citation not in known]
