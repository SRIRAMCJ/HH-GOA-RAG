from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860/api/query")
    parser.add_argument("--queries", default="tests/benchmark_queries.json")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    latencies: list[float] = []
    results = []
    for query in queries:
        started = time.perf_counter()
        try:
            response = requests.post(args.url, json={"query": query, "debug": True}, timeout=30)
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            body = response.json()
            latencies.append(float(body.get("latency_ms", elapsed)))
            results.append({"query": query, "ok": True, "wall_ms": elapsed, "pipeline_ms": body.get("latency_ms"), "stages_ms": body.get("stages_ms", {})})
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            results.append({"query": query, "ok": False, "wall_ms": elapsed, "error": str(exc)})

    summary = {
        "count": len(queries),
        "successful": len(latencies),
        "failure_rate": 1 - (len(latencies) / max(len(queries), 1)),
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "p50_ms": percentile(latencies, 0.50),
        "p70_ms": percentile(latencies, 0.70),
        "p100_ms": max(latencies) if latencies else None,
        "under_200ms": bool(latencies) and max(latencies) < 200,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
