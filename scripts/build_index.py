"""Build a reproducible RAG index from AI4Bharat MSMARCO-XI data.

The dataset is never committed to GitHub. The builder runs on cloud/CI
infrastructure, creates four chunking variants, and writes a portable FAISS +
BM25 artifact bundle.

MSMARCO-XI publishes language-specific JSONL files. We download the requested
file directly from the dataset repository instead of relying on the deprecated
loading-script configuration mechanism in newer `datasets` releases.
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
from huggingface_hub import hf_hub_download
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
    """Create the four required chunking variants.

    `semantic` is deliberately deterministic for the indexing benchmark: it
    groups adjacent sentences into larger topical windows without requiring a
    second embedding pass during ingestion.
    """
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


def dataset_filename(config: str, split: str) -> str:
    if split == "train":
        suffix = "train"
    elif split in {"validation", "val"}:
        suffix = "val"
    else:
        raise ValueError("MSMARCO-XI supports train/validation files for this builder")
    return f"{split if split == 'train' else 'validation'}/{config}{suffix}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ta", help="MSMARCO-XI language code, e.g. ta, hi, bn")
    parser.add_argument("--split", default="train", choices=["train", "validation", "val"])
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    args = parser.parse_args()

    if args.max_docs <= 0:
        raise ValueError("--max-docs must be greater than zero")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    filename = dataset_filename(args.config, args.split)
    dataset_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        token=None,
    )
    print(f"Using dataset file: {filename}")

    chunks: list[Chunk] = []
    with open(dataset_path, "r", encoding="utf-8") as source:
        for idx, line in enumerate(source):
            if idx >= args.max_docs:
                break
            if not line.strip():
                continue

            row = json.loads(line)
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
                    "answer": normalize_text(row.get("Answer")),
                    "english_answer": normalize_text(row.get("Eng_Answer")),
                    "query_type": normalize_text(row.get("query_type")),
                    "passage_id": passage_id,
                }
                chunks.extend(make_chunks(f"{query_id}:{passage_id}", text, metadata))

    if not chunks:
        raise RuntimeError("No passages were found. Verify the language code and split.")

    texts = [c.text for c in chunks]
    print(f"Created {len(chunks)} chunks from {min(args.max_docs, idx + 1)} dataset rows")

    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
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
        "source_file": filename,
        "config": args.config,
        "split": args.split,
        "documents_indexed": min(args.max_docs, idx + 1),
        "chunks": len(chunks),
        "embedding_model": args.embedding_model,
        "dimension": int(matrix.shape[1]),
        "strategies": sorted({c.strategy for c in chunks}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
