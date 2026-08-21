"""Local-only knowledge loader with explicit file and size allow-lists."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from rag_eval.models import Document

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
SUPPORTED_SUFFIXES = {".md", ".txt"}
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class UnsafePathError(ValueError):
    """Raised when a requested path escapes the configured knowledge root."""


class KnowledgeLoader:
    def __init__(self, root: str | Path, max_bytes: int = MAX_DOCUMENT_BYTES) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes

    def _validate(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise UnsafePathError(f"Path escapes knowledge root: {path}")
        if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {resolved.suffix}")
        if resolved.stat().st_size > self.max_bytes:
            raise ValueError(f"Document exceeds {self.max_bytes} bytes: {resolved.name}")
        return resolved

    def load_file(self, path: str | Path) -> Document:
        resolved = self._validate(Path(path))
        raw = resolved.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        match = FRONT_MATTER.match(raw)
        if match:
            parsed = yaml.safe_load(match.group(1)) or {}
            if not isinstance(parsed, dict):
                raise ValueError(f"Front matter must be a mapping: {resolved.name}")
            metadata = parsed
            raw = raw[match.end() :]
        document_id = str(metadata.pop("document_id", resolved.stem.upper().replace("_", "-")))
        title = str(metadata.pop("title", self._extract_title(raw, resolved.stem)))
        return Document(
            document_id=document_id,
            title=title,
            content=self._normalise(raw),
            source_path=str(resolved.relative_to(self.root)),
            metadata=metadata,
        )

    def load_all(self) -> list[Document]:
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        return [self.load_file(path) for path in paths]

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return fallback.replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _normalise(content: str) -> str:
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in content.splitlines()]
        return "\n".join(lines).strip()
