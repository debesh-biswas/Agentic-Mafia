"""Text embedders. Both return L2-normalized float32 vectors."""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()  # one embedder may be shared by several stores/threads

    def embed(self, text: str) -> np.ndarray:
        k = _key(text)
        with self._lock:
            if k not in self._cache:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # optional extra: pip install -e ".[memory]"

                    self._model = SentenceTransformer(self.model_name)
                v = np.asarray(self._model.encode(text, normalize_embeddings=True), dtype=np.float32)
                self._cache[k] = v / (np.linalg.norm(v) or 1.0)
            return self._cache[k]


class FakeEmbedder:
    """Deterministic bag-of-words embedder: each word hashes to a dim, so texts sharing words are similar."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()) or [text]:
            v[int(_key(w)[:8], 16) % self.dim] += 1.0
        return v / (np.linalg.norm(v) or 1.0)
