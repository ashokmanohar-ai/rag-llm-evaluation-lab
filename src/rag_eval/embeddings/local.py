"""Local embeddings with an explicit deterministic CI fallback."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedding:
    """Stable feature hashing; a test fallback, not a semantic model."""

    model_name = "deterministic-hashing-v1"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimensions
                matrix[row, index] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedding:
    """Adapter for the preferred all-MiniLM-L6-v2 offline model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the ml extra: pip install -e '.[ml]'") from exc
        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        values = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(values, dtype=np.float64)


def build_embedding_provider(
    config: dict[str, Any],
) -> HashingEmbedding | SentenceTransformerEmbedding:
    provider = str(config.get("provider", "auto"))
    model = str(config.get("model", "all-MiniLM-L6-v2"))
    if provider in {"sentence_transformers", "auto"}:
        try:
            return SentenceTransformerEmbedding(model)
        except (RuntimeError, OSError):
            if provider != "auto":
                raise
            LOGGER.warning(
                "sentence-transformers is unavailable; using deterministic hashing fallback"
            )
    return HashingEmbedding(int(config.get("dimensions", 384)))
