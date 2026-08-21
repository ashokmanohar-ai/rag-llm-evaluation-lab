"""Embedding protocol."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> NDArray[np.float64]: ...
