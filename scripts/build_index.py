"""Build a reproducible RAG index from streamed AI4Bharat MSMARCO-XI data.

The dataset is never committed to GitHub. The builder runs on cloud/CI
infrastructure, creates four chunking variants, and writes a portable FAISS +
BM25 artifact bundle.
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
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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
    return re.sub(r"\s+", " ", str(value)).strip()


def sentence_split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?।])\s+", text) if p.strip()]


def fixed_chunks(text: str, words: int = 120, overlap: int = 30) -> Iterable[str]:
    tokens = text.split()
    step = max(1, words - overlap)
    for start in range(0, len(tokens), step):
        body = " ".join(tokens[start:start + words]).strip()
        if body:
            yield body
        if start + words >= len(tokens):
            break


def make_chunks(document_id: str, text: str, metadata: dict[str, Any]) -> list[Chunk]:
    sents = sentence_split(text)
    chunks: list[Chunk] = []
    for i in range(0, len(sents), 5):
        body = " ".join(sents[i:i + 5])
        if body:
            chunks.append(Chunk(f"{document_id}:sentence:{i}", document_id, "sentence", body, metadata))
    for i, body in enumerate(fixed_chunks(text)):
        chunks.append(Chunk(f"{document_id}:sliding:{i}", document_id, "sliding_window", body, metadata))
    if text:
        chunks.append(Chunk(f"{document_id}:metadata:0", document_id, "metadata_aware", text[:1800], metadata))
    for i in range(0, len(sents), 8):
        body = " ".join(sents[i:i + 8])
        if body:
            chunks.append(Chunk(f"{document_id}:semantic:{i}", document_id, "semantic", body, metadata))
    return chunks


def passage_texts(row: dict[str, Any]) -> list[tuple[str, str]]:
    passages = row.get("passages") or {}
    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    out: list[tuple[str, str]] = []
    for i, value in enumerate(translated):
        text = normalize_text(value)
        if text:
            out.append((f"translated:{i}", text))
    for i, value in enumerate(english):
        text = normalize_text(value)
        if text:
            out.append((f"english:{i}", text))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ta", help="MSMARCO-XI language config, e.g. ta, hi, en-compatible content")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID, args.config, split=args.split, streaming=True)
    chunks: list[Chunk] = []

    for idx, raw in enumerate(ds):
        if idx >= args.max_docs:
            break
        row = dict(raw)
        query_id = normalize_text(row.get("query_id") or idx)
        target_lang = normalize_text(row.get("target_lang") or args.config)
        for passage_id, text in passage_texts(row):
            metadata = {
                "dataset": DATASET_ID,
                "config": args.config,
                "target_lang": target_lang,
                "query_id": query_id,
                "query": normalize_text(row.get("query")),
                "english_query": normalize_text(row.get("Eng_Query")),
                "passage_id": passage_id,
            }
            chunks.extend(make_chunks(f"{query_id}:{passage_id}", text, metadata))

    if not chunks:
        raise RuntimeError("No passages were found. Verify the config/split with inspect_dataset.py.")

    texts = [c.text for c in chunks]
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    matrix = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(out / "vectors.faiss"))

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    with (out / "bm25_tokens.jsonl").open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps(text.lower().split(), ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_ID,
        "config": args.config,
        "split": args.split,
        "documents": args.max_docs,
        "chunks": len(chunks),
        "embedding_model": args.embedding_model,
        "dimension": int(matrix.shape[1]),
        "strategies": sorted({c.strategy for c in chunks}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
