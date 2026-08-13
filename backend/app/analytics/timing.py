from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timed(stages: dict[str, float], name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        stages[name] = (perf_counter() - start) * 1000.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
