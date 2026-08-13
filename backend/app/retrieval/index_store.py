from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from app.retrieval.hybrid import IndexedChunk, RetrievedChunk


class DenseIndex:
    def __init__(self, index: faiss.Index, chunks: Sequence[IndexedChunk]):
        self.index = index
        self.chunks = list(chunks)

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex | None":
        root = Path(directory)
        index_path = root / "faiss.index"
        chunks_path = root / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            return None
        index = faiss.read_index(str(index_path))
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [IndexedChunk(**item) for item in raw]
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index/chunk metadata size mismatch")
        return cls(index, chunks)

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: Sequence[IndexedChunk]) -> "DenseIndex":
        if len(chunks) == 0:
            raise ValueError("Cannot build an empty dense index")
        vectors = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks)

    def search(self, vector: np.ndarray, top_k: int = 20) -> list[RetrievedChunk]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, ids = self.index.search(query, min(top_k, len(self.chunks)))
        return [
            RetrievedChunk(self.chunks[i], float(score), "dense")
            for score, i in zip(scores[0], ids[0])
            if i >= 0
        ]
