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
        index_path = root / "vectors.faiss"
        chunks_path = root / "chunks.jsonl"
        if not index_path.exists() or not chunks_path.exists():
            index_path = root / "faiss.index"
            chunks_path = root / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            return None
        index = faiss.read_index(str(index_path))
        if chunks_path.suffix == ".jsonl":
            raw = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [IndexedChunk(
            chunk_id=str(item["chunk_id"]),
            document_id=str(item["document_id"]),
            text=str(item["text"]),
            strategy=str(item.get("strategy", "unknown")),
            metadata=item.get("metadata") or {},
        ) for item in raw]
        if index.ntotal != len(chunks):
            raise ValueError(f"FAISS index/chunk mismatch: {index.ntotal} != {len(chunks)}")
        return cls(index, chunks)

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: Sequence[IndexedChunk]) -> "DenseIndex":
        if not chunks:
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
        return [RetrievedChunk(self.chunks[i], float(score), "dense") for score, i in zip(scores[0], ids[0]) if i >= 0]
