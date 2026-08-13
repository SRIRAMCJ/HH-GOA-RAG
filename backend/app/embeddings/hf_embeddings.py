from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from huggingface_hub import InferenceClient


class HFEmbedder:
    """Hosted feature-extraction client; no model weights are loaded locally."""

    def __init__(self, token: str, model: str):
        self.client = InferenceClient(api_key=token)
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for text in texts:
            result = self.client.feature_extraction(text, model=self.model)
            arr = np.asarray(result, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            vectors.append(arr)
        matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
