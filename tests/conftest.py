from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from rag_eval.config import ROOT, load_rag_config, load_yaml
from rag_eval.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def offline_config() -> dict[str, Any]:
    config = copy.deepcopy(load_rag_config())
    config["embeddings"]["provider"] = "hashing"
    config["generation"]["provider"] = "mock"
    config["reranking"]["backend"] = "lexical"
    return config


@pytest.fixture(scope="session")
def pipeline(project_root: Path, offline_config: dict[str, Any]) -> RAGPipeline:
    return RAGPipeline(project_root / "data/knowledge", offline_config)


@pytest.fixture(scope="session")
def gate_config(project_root: Path) -> dict[str, Any]:
    return load_yaml(project_root / "config/quality-gates.yaml")
