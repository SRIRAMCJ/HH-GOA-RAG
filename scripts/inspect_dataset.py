"""Inspect the official AI4Bharat MSMARCO-XI dataset files.

MSMARCO-XI publishes one JSONL train/validation file per language. This
inspector uses the Hub file API directly so it works with current versions of
`datasets` even when the legacy dataset loading script is not executed.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from huggingface_hub import hf_hub_download

DATASET_ID = "ai4bharat/MSMARCO-XI"
LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def compact(value: Any, limit: int = 500) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: compact(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v, limit) for v in value[:20]]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(LANGUAGES), default="ta")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    suffix = "train" if args.split == "train" else "val"
    folder = "train" if args.split == "train" else "validation"
    filename = f"{folder}/{args.config}{suffix}.jsonl"

    print("DATASET:", DATASET_ID)
    print("LANGUAGES:", json.dumps(LANGUAGES, ensure_ascii=False))
    print("SELECTED:", args.config, LANGUAGES[args.config], args.split)
    print("FILE:", filename)

    path = hf_hub_download(repo_id=DATASET_ID, filename=filename, repo_type="dataset", token=None)
    with open(path, "r", encoding="utf-8") as source:
        for i, line in enumerate(source):
            if not line.strip():
                continue
            print(json.dumps(compact(json.loads(line)), ensure_ascii=False, indent=2, default=str))
            if i + 1 >= args.rows:
                break


if __name__ == "__main__":
    main()
