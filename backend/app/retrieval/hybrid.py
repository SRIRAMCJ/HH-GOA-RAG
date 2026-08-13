from dataclasses import dataclass
from typing import Any, Sequence

from rank_bm25 import BM25Okapi


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    text: str
    strategy: str
    metadata: dict | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk: IndexedChunk
    score: float
    source: str


class HybridRetriever:
    """BM25 + FAISS dense retrieval with reciprocal-rank fusion."""

    def __init__(self, chunks: Sequence[IndexedChunk], dense_index: Any | None = None):
        self.chunks = list(chunks)
        self.dense_index = dense_index
        self._bm25 = BM25Okapi([c.text.lower().split() for c in self.chunks]) if self.chunks else None

    def lexical_search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(self.chunks[i], float(score), "bm25") for i, score in ranked]

    def dense_search(self, vector, top_k: int = 20) -> list[RetrievedChunk]:
        if self.dense_index is None:
            return []
        return self.dense_index.search(vector, top_k)

    def fuse(self, dense: Sequence[RetrievedChunk], lexical: Sequence[RetrievedChunk], top_k: int = 20) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        objects: dict[str, RetrievedChunk] = {}
        for results in (dense, lexical):
            for rank, item in enumerate(results, start=1):
                fused[item.chunk.chunk_id] = fused.get(item.chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
                objects[item.chunk.chunk_id] = item
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(objects[cid].chunk, score, "hybrid") for cid, score in ordered]
