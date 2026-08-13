from __future__ import annotations

from typing import Sequence

import numpy as np
from huggingface_hub import InferenceClient

from app.retrieval.hybrid import RetrievedChunk


class HostedReranker:
    """Batched hosted semantic reranker; no local checkpoint is loaded."""

    def __init__(self, token: str | None, model: str):
        self.client = InferenceClient(api_key=token) if token else None
        self.model = model

    @staticmethod
    def _pool(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        arr = arr.reshape(-1)
        return arr / max(float(np.linalg.norm(arr)), 1e-12)

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates or not self.client:
            return list(candidates[:top_k])
        try:
            q = self._pool(self.client.feature_extraction(query, model=self.model))
            raw = self.client.feature_extraction([x.chunk.text for x in candidates], model=self.model)
            scored = []
            for item, vector in zip(candidates, raw):
                v = self._pool(vector)
                scored.append(RetrievedChunk(item.chunk, float(np.dot(q, v)), "semantic_reranker"))
            return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]
        except Exception:
            return list(candidates[:top_k])
