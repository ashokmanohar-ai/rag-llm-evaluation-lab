"""Configuration loading with environment overrides and safe defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {resolved}")
    return value


def load_rag_config(path: str | Path = "config/rag.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    generation = config.setdefault("generation", {})
    embeddings = config.setdefault("embeddings", {})
    reranking = config.setdefault("reranking", {})
    generation["provider"] = os.getenv("LLM_PROVIDER", generation.get("provider", "mock"))
    embeddings["provider"] = os.getenv("EMBEDDING_PROVIDER", embeddings.get("provider", "auto"))
    embeddings["model"] = os.getenv("EMBEDDING_MODEL", embeddings.get("model", "all-MiniLM-L6-v2"))
    if "RERANKER_ENABLED" in os.environ:
        reranking["enabled"] = os.environ["RERANKER_ENABLED"].lower() in {"1", "true", "yes"}
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
