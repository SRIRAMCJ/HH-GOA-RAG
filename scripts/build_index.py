"""Build a reproducible RAG index from AI4Bharat MSMARCO-XI data.

The dataset is never committed to GitHub. The builder runs on cloud/CI,
downloads the requested language Parquet file to the runner, reads it with
PyArrow in batches, creates a practical retrieval chunk set, and writes a
portable FAISS + BM25 artifact bundle.

The default build intentionally uses one sliding-window strategy. The older
four-strategy mode multiplies the number of passages and CPU embedding work
and made GitHub-hosted CPU builds impractically slow. Use --all-strategies
only for offline benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

DATASET_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANGUAGE_FILE_STEMS = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
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


def make_chunks(
    document_id: str,
    text: str,
    metadata: dict[str, Any],
    all_strategies: bool = False,
) -> list[Chunk]:
    """Create practical chunks for retrieval.

    Sliding-window chunks are the default because they give predictable
    coverage at a fraction of the embedding cost of four parallel variants.
    """
    chunks: list[Chunk] = []
    for i, body in enumerate(fixed_chunks(text)):
        chunks.append(Chunk(f"{document_id}:sliding:{i}", document_id, "sliding_window", body, metadata))

    if not all_strategies:
        return chunks

    sents = sentence_split(text)
    for i in range(0, len(sents), 5):
        body = " ".join(sents[i:i + 5])
        if body:
            chunks.append(Chunk(f"{document_id}:sentence:{i}", document_id, "sentence", body, metadata))

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


def iter_rows(config: str, split: str, max_docs: int):
    """Download the Parquet file once and read only the required rows in batches."""
    filename = dataset_filename(config, split)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to download MSMARCO-XI from Hugging Face")

    print(f"[1/5] Downloading dataset file: {filename}", flush=True)
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    file_size = Path(parquet_path).stat().st_size
    print(f"[2/5] Local Parquet: {file_size / (1024 ** 3):.2f} GiB", flush=True)

    parquet_file = pq.ParquetFile(parquet_path)
    print(f"[3/5] Parquet rows={parquet_file.metadata.num_rows}, row_groups={parquet_file.metadata.num_row_groups}", flush=True)

    yielded = 0
    for batch_number, batch in enumerate(parquet_file.iter_batches(batch_size=1000), start=1):
        rows = batch.to_pylist()
        for row in rows:
            if yielded >= max_docs:
                return
            yield row
            yielded += 1
        if batch_number % 5 == 0 or yielded >= max_docs:
            print(f"[4/5] Read {yielded}/{max_docs} dataset rows", flush=True)


def write_chunk(fh, chunk: Chunk) -> None:
    fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ta", help="MSMARCO-XI language code, e.g. ta, hi, bn")
    parser.add_argument("--split", default="train", choices=["train", "validation", "val"])
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--output", default="artifacts/index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--all-strategies", action="store_true")
    args = parser.parse_args()

    if args.max_docs <= 0:
        raise ValueError("--max-docs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    filename = dataset_filename(args.config, args.split)
    chunks_path = out / "chunks.jsonl"
    bm25_path = out / "bm25_tokens.jsonl"

    print("Starting cloud index build", flush=True)
    print(f"Config={args.config} split={args.split} max_docs={args.max_docs}", flush=True)
    print(f"Chunk mode={'all strategies' if args.all_strategies else 'sliding_window'}", flush=True)

    model = None
    index = None
    chunk_count = 0
    rows_read = 0

    with chunks_path.open("w", encoding="utf-8") as chunks_fh, bm25_path.open("w", encoding="utf-8") as bm25_fh:
        pending_chunks: list[Chunk] = []

        for idx, row in enumerate(iter_rows(args.config, args.split, args.max_docs)):
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
                pending_chunks.extend(make_chunks(
                    f"{query_id}:{passage_id}", text, metadata, args.all_strategies
                ))

            # Process embeddings incrementally so the full chunk corpus is
            # never held in RAM and progress is visible in GitHub Actions.
            if len(pending_chunks) >= args.batch_size * 4:
                if model is None:
                    print("[5/5] Loading embedding model...", flush=True)
                    model = SentenceTransformer(args.embedding_model)
                texts = [c.text for c in pending_chunks]
                print(f"Embedding {len(texts)} chunks (rows processed={rows_read})...", flush=True)
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                )
                matrix = np.asarray(embeddings, dtype="float32")
                if index is None:
                    index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                for chunk in pending_chunks:
                    write_chunk(chunks_fh, chunk)
                    bm25_fh.write(json.dumps(chunk.text.lower().split(), ensure_ascii=False) + "\n")
                chunk_count += len(pending_chunks)
                pending_chunks.clear()
                print(f"Embedded/indexed {chunk_count} chunks", flush=True)

        if pending_chunks:
            if model is None:
                print("[5/5] Loading embedding model...", flush=True)
                model = SentenceTransformer(args.embedding_model)
            texts = [c.text for c in pending_chunks]
            print(f"Embedding final {len(texts)} chunks...", flush=True)
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=args.batch_size,
                convert_to_numpy=True,
            )
            matrix = np.asarray(embeddings, dtype="float32")
            if index is None:
                index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            for chunk in pending_chunks:
                write_chunk(chunks_fh, chunk)
                bm25_fh.write(json.dumps(chunk.text.lower().split(), ensure_ascii=False) + "\n")
            chunk_count += len(pending_chunks)

    if index is None or chunk_count == 0:
        raise RuntimeError("No passages/chunks were found. Verify the language code and split.")

    faiss.write_index(index, str(out / "vectors.faiss"))

    manifest = {
        "dataset": DATASET_ID,
        "config": args.config,
        "split": args.split,
        "source_file": filename,
        "rows_read": rows_read,
        "chunks": chunk_count,
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(index.d),
        "strategies": sorted({"sliding_window", "sentence", "metadata_aware", "semantic"} if args.all_strategies else {"sliding_window"}),
        "index_type": "faiss.IndexFlatIP",
        "normalized_embeddings": True,
        "retrieval": ["dense_faiss", "bm25", "rrf"],
        "build_mode": "incremental_cpu_embedding",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("INDEX BUILD COMPLETE", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
