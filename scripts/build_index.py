"""Build a reproducible RAG index from AI4Bharat MSMARCO-XI data.

The dataset is never committed to GitHub. The builder runs on cloud/CI,
streams the requested language Parquet file, creates four chunking variants,
and writes a portable FAISS + BM25 artifact bundle.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
import faiss
import numpy as np
from huggingface_hub import hf_hub_url
from sentence_transformers import SentenceTransformer

DATASET_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# The current MSMARCO-XI Hub repository stores the large source files as
# language-specific Parquet files. The filename stem is not always identical
# to the two-letter language code (e.g. Gujarati -> gujtrain.parquet,
# Tamil -> tamtrain.parquet).
LANGUAGE_FILE_STEMS = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}


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
    """Create four deterministic chunking variants for evaluation."""
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

    # Deterministic semantic proxy: sentence-group windows. The production
    # benchmark can compare this against a true embedding-based semantic
    # splitter without changing the index schema.
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
    try:
        stem = LANGUAGE_FILE_STEMS[config]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_FILE_STEMS))
        raise ValueError(f"Unsupported MSMARCO-XI language code '{config}'. Supported: {supported}") from exc

    if split == "train":
        return f"train/{stem}train.parquet"
    if split in {"validation", "val"}:
        return f"validation/{stem}val.parquet"
    raise ValueError("MSMARCO-XI supports train/validation files")


def stream_rows(config: str, split: str, max_docs: int):
    """Stream Parquet rows from Hugging Face without downloading the full file."""
    filename = dataset_filename(config, split)
    url = hf_hub_url(DATASET_ID, filename=filename, repo_type="dataset")
    token = __import__("os").getenv("HF_TOKEN")

    dataset = load_dataset(
        "parquet",
        data_files={"data": url},
        split="data",
        streaming=True,
        token=token,
    )

    for idx, row in enumerate(dataset):
        if idx >= max_docs:
            break
        yield row


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
    print(f"Streaming dataset file: {filename}")

    chunks: list[Chunk] = []
    rows_read = 0
    for idx, row in enumerate(stream_rows(args.config, args.split, args.max_docs)):
        rows_read = idx + 1
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
    print(f"Created {len(chunks)} chunks from {rows_read} dataset rows")

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

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    tokenized = [c.text.lower().split() for c in chunks]
    with (out / "bm25_tokens.jsonl").open("w", encoding="utf-8") as fh:
        for tokens in tokenized:
            fh.write(json.dumps(tokens, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_ID,
        "config": args.config,
        "split": args.split,
        "source_file": filename,
        "rows_read": rows_read,
        "chunks": len(chunks),
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(matrix.shape[1]),
        "strategies": sorted({c.strategy for c in chunks}),
        "index_type": "faiss.IndexFlatIP",
        "normalized_embeddings": True,
        "retrieval": ["dense_faiss", "bm25", "rrf"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
