from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.ingestion import Chunker, KnowledgeLoader
from rag_eval.ingestion.loader import UnsafePathError
from rag_eval.models import Document


def test_loader_reads_original_documents_and_metadata(project_root: Path) -> None:
    documents = KnowledgeLoader(project_root / "data/knowledge").load_all()
    annual = next(item for item in documents if item.document_id == "HR-ANNUAL-LEAVE")
    assert len(documents) >= 20
    assert annual.metadata["region"] == "UK"
    assert annual.metadata["status"] == "CURRENT"
    assert "25 days" in annual.content


def test_loader_rejects_path_traversal(project_root: Path) -> None:
    loader = KnowledgeLoader(project_root / "data/knowledge")
    with pytest.raises(UnsafePathError):
        loader.load_file(project_root / "README.md")


def test_loader_rejects_unsupported_file(tmp_path: Path) -> None:
    target = tmp_path / "unsafe.html"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        KnowledgeLoader(tmp_path).load_file(target)


def test_loader_rejects_oversized_document(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_text("123456", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        KnowledgeLoader(tmp_path, max_bytes=5).load_file(target)


def test_fixed_chunking_has_stable_ids_and_overlap() -> None:
    document = Document(
        document_id="DOC",
        title="Doc",
        content="alpha " * 50,
        source_path="doc.txt",
    )
    chunks = Chunker("fixed", 100, 20).chunk_document(document)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "DOC-C001"
    assert chunks[1].content.startswith(chunks[0].content[-20:])


def test_recursive_chunking_preserves_heading_and_metadata() -> None:
    document = Document(
        document_id="DOC",
        title="Doc",
        content="# One\n\nFirst paragraph.\n\n## Two\n\nSecond paragraph.",
        source_path="doc.md",
        metadata={"region": "UK"},
    )
    chunks = Chunker("recursive", 100, 10).chunk_document(document)
    assert [chunk.section for chunk in chunks] == ["One", "Two"]
    assert chunks[0].metadata["region"] == "UK"


@pytest.mark.parametrize(
    ("strategy", "size", "overlap"),
    [("unknown", 100, 10), ("fixed", 49, 10), ("fixed", 100, 100)],
)
def test_invalid_chunk_configuration_is_rejected(strategy: str, size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        Chunker(strategy, size, overlap)
