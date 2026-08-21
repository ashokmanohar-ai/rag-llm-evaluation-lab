"""Fixed and heading-aware recursive chunking."""

from __future__ import annotations

import re

from rag_eval.models import Chunk, Document


class Chunker:
    def __init__(self, strategy: str = "recursive", chunk_size: int = 600, overlap: int = 100):
        if chunk_size < 50:
            raise ValueError("chunk_size must be at least 50 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        if strategy not in {"fixed", "recursive"}:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk_document(document)]

    def chunk_document(self, document: Document) -> list[Chunk]:
        pieces = (
            self._fixed(document.content)
            if self.strategy == "fixed"
            else self._recursive(document.content)
        )
        chunks: list[Chunk] = []
        for number, (section, content) in enumerate(pieces, start=1):
            metadata = {
                **document.metadata,
                "document_id": document.document_id,
                "document_title": document.title,
                "section": section,
                "chunk_id": f"{document.document_id}-C{number:03d}",
                "source_path": document.source_path,
            }
            chunks.append(
                Chunk(
                    chunk_id=metadata["chunk_id"],
                    document_id=document.document_id,
                    document_title=document.title,
                    section=section,
                    content=content.strip(),
                    source_path=document.source_path,
                    metadata=metadata,
                )
            )
        return chunks

    def _fixed(self, content: str) -> list[tuple[str, str]]:
        step = self.chunk_size - self.overlap
        return [
            ("document", content[start : start + self.chunk_size])
            for start in range(0, len(content), step)
        ]

    def _recursive(self, content: str) -> list[tuple[str, str]]:
        sections = self._sections(content)
        output: list[tuple[str, str]] = []
        for heading, body in sections:
            if len(body) <= self.chunk_size:
                if body.strip():
                    output.append((heading, body))
                continue
            output.extend((heading, value) for value in self._split_with_overlap(body))
        return output

    def _split_with_overlap(self, text: str) -> list[str]:
        paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            tail = current[-self.overlap :] if self.overlap and current else ""
            current = f"{tail}\n\n{paragraph}".strip()
            while len(current) > self.chunk_size:
                chunks.append(current[: self.chunk_size])
                current = current[self.chunk_size - self.overlap :]
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _sections(content: str) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        heading = "document"
        buffer: list[str] = []
        for line in content.splitlines():
            match = re.match(r"^#{1,6}\s+(.+)$", line)
            if match:
                if buffer:
                    result.append((heading, "\n".join(buffer).strip()))
                heading = match.group(1).strip()
                buffer = []
            else:
                buffer.append(line)
        if buffer:
            result.append((heading, "\n".join(buffer).strip()))
        return result or [("document", content)]
