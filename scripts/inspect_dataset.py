"""Inspect the official HH Goa MSMARCO-XI dataset without downloading it in full.

Usage:
  python scripts/inspect_dataset.py --rows 5

The script uses Hugging Face streaming so the developer machine never needs the
full dataset. It prints configs/splits and a few normalized example records.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

DATASET_ID = "ai4bharat/MSMARCO-XI"


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
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    configs = get_dataset_config_names(DATASET_ID)
    print("DATASET:", DATASET_ID)
    print("CONFIGS:", json.dumps(configs, ensure_ascii=False))

    for config in configs:
        splits = get_dataset_split_names(DATASET_ID, config)
        print(f"CONFIG={config} SPLITS={splits}")

    # Use the first available config/split only for a tiny schema/sample inspection.
    config = configs[0]
    split = get_dataset_split_names(DATASET_ID, config)[0]
    stream = load_dataset(DATASET_ID, config, split=split, streaming=True)
    print("SELECTED:", config, split)

    for i, row in enumerate(stream):
        print(json.dumps(compact(row), ensure_ascii=False, indent=2, default=str))
        if i + 1 >= args.rows:
            break


if __name__ == "__main__":
    main()
