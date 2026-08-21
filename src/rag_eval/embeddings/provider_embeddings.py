"""Reserved adapter contract for managed embedding APIs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class CallableEmbeddingProvider:
    """Wrap an approved provider SDK function without coupling core evaluation to it."""

    def __init__(self, model_name: str, embed_fn: Callable[[list[str]], list[list[float]]]):
        self.model_name = model_name
        self._embed_fn = embed_fn

    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        return np.asarray(self._embed_fn(texts), dtype=np.float64)
