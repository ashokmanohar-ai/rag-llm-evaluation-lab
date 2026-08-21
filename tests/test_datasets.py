from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_evaluation_corpus_has_meaningful_target_size(project_root: Path) -> None:
    root = project_root / "data/evaluation"
    counts = {path.name: len(_load(path)) for path in root.glob("*.jsonl")}
    assert counts == {
        "retrieval.jsonl": 25,
        "qa.jsonl": 25,
        "hallucination.jsonl": 10,
        "citations.jsonl": 10,
        "adversarial.jsonl": 10,
    }
    assert sum(counts.values()) == 80


def test_every_case_has_unique_id(project_root: Path) -> None:
    records = [
        record
        for path in (project_root / "data/evaluation").glob("*.jsonl")
        for record in _load(path)
    ]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
