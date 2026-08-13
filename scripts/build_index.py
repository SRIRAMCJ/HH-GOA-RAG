"""Build a compact, reproducible RAG index from streamed MSMARCO-XI data.

The full dataset is intentionally never committed to GitHub. The script streams
examples, creates several chunking variants, embeds them, and writes a local
FAISS/BM25 bundle. On cloud infrastructure this can be run as a build job.

This first version is deliberately bounded by --max-docs so latency experiments
can start quickly. Increase the bound on a remote runner after validating the
pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATASET_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    strategy: str
    text: str
    metadata: dict[str, Any]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def fixed_chunks(text: str, words: int = 120, overlap: int = 30) -> Iterable[str]:
    tokens = text.split()
    if not tokens:
        return
    step = max(1, words - overlap)
    for start in range(0, len(tokens), step):
        chunk = " ".join(tokens[start : start + words])
        if chunk:
            yield chunk
        if start + words >= len(tokens):
            break


def make_chunks(document_id: str, text: str, metadata: dict[str, Any]) -> list[Chunk]:
    sentences = sentence_split(text)
    chunks: list[Chunk] = []

    # Strategy 1: sentence groups — preserves linguistic boundaries.
    for i in range(0, len(sentences), 5):
        body = " ".join(sentences[i : i + 5])
        if body:
            chunks.append(Chunk(f"{document_id}:sent:{i}", document_id, "sentence", body, metadata))

    # Strategy 2: sliding window — overlap improves boundary recall.
    for i, body in enumerate(fixed_chunks(text)):
        chunks.append(Chunk(f"{document_id}:slide:{i}", document_id, "sliding_window", body, metadata))

    # Strategy 3: metadata-aware — keep the original document metadata attached.
    # The retrieval layer can use this to filter/rerank by language/source fields.
    if text:
        chunks.append(Chunk(f"{document_id}:metadata:0", document_id, "metadata_aware", text[:1800], metadata))

    # Strategy 4: semantic slot — sentence-group boundaries are the safe baseline;
    # a semantic-boundary pass can replace/augment this without changing the index API.
    for i in range(0, len(sentences), 8):
        body = " ".join(sentences[i : i + 8])
        if body:
            chunks.append(Chunk(f"{document_id}:semantic:{i}", document_id, "semantic", body, metadata))

    return chunks


def choose_text(row: dict[str, Any]) -> tuple[str, str]:
    """Best-effort normalization across MSMARCO-XI schema variants."""
    candidates = ["passage", "text", "context", "document", "content", "answer"]
    for key in candidates:
        value = normalize_text(row.get(key))
        if value:
            return value, key
    # Fallback: concatenate useful string fields, excluding identifiers.
    pieces = []
    for key, value in row.items():
        if key.lower() in {"id", "query_id", "document_id"}:
            continue
        value = normalize_text(value)
        if value:
            pieces.append(f"{key}: {value}")
    return " ".join(pieces), "composite"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID, args.config, split=args.split, streaming=True)
    chunks: list[Chunk] = []

    for idx, row in enumerate(ds):
        if idx >= args.max_docs:
            break
        row = dict(row)
        text, source_field = choose_text(row)
        if not text:
            continue
        document_id = normalize_text(row.get("document_id") or row.get("id") or idx)
        metadata = {
            "source_field": source_field,
            "language": row.get("language") or row.get("lang"),
            "query_id": row.get("query_id"),
            "dataset": DATASET_ID,
        }
        chunks.extend(make_chunks(document_id, text, metadata))

    if not chunks:
        raise RuntimeError("No usable text found. Run scripts/inspect_dataset.py to inspect the schema.")

    texts = [c.text for c in chunks]
    model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(out / "vectors.faiss"))

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    # BM25 is reconstructed at runtime from these token lists; keeping the index
    # representation JSON makes the artifact portable and inspectable.
    with (out / "bm25_tokens.jsonl").open("w", encoding="utf-8") as f:
        for tokens in tokenized:
            f.write(json.dumps(tokens, ensure_ascii=False) + "\n")

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET_ID,
                "config": args.config,
                "split": args.split,
                "documents": args.max_docs,
                "chunks": len(chunks),
                "embedding_model": args.embedding_model,
                "dimension": int(matrix.shape[1]),
                "strategies": sorted({c.strategy for c in chunks}),
            },
            f,
            indent=2,
        )

    print(f"Built {len(chunks)} chunks from up to {args.max_docs} documents")
    print(f"Artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
