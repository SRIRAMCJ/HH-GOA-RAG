from __future__ import annotations

import math
from typing import Sequence

from huggingface_hub import InferenceClient

from app.retrieval.hybrid import RetrievedChunk


class HostedReranker:
    """Uses a hosted HF text-ranking model when available.

    If the provider does not expose the selected model, RRF order is retained
    instead of making the request path fail.
    """

    def __init__(self, token: str | None, model: str):
        self.client = InferenceClient(api_key=token) if token else None
        self.model = model

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if not self.client:
            return list(candidates[:top_k])
        try:
            pairs = [(query, item.chunk.text) for item in candidates]
            result = self.client.text_classification(pairs, model=self.model)
            # Providers differ in output shape. Only accept scalar-like scores.
            scored = []
            for item, raw in zip(candidates, result):
                if isinstance(raw, list):
                    values = [x.get("score", 0.0) for x in raw if isinstance(x, dict)]
                    score = max(values) if values else 0.0
                elif isinstance(raw, dict):
                    score = float(raw.get("score", 0.0))
                else:
                    score = float(raw)
                scored.append(RetrievedChunk(item.chunk, score, "reranker"))
            return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]
        except Exception:
            return list(candidates[:top_k])
