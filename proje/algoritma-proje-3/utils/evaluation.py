from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RunResult:
    tour_length: float
    runtime_sec: float
    seed: int


@dataclass
class AggregateStats:
    best: float
    mean: float
    std: float
    min_runtime: float
    mean_runtime: float
    max_runtime: float


def compute_aggregate_stats(results: Iterable[RunResult]) -> AggregateStats:
    lens: List[float] = [r.tour_length for r in results]
    times: List[float] = [r.runtime_sec for r in results]
    arr_len = np.array(lens, dtype=float)
    arr_time = np.array(times, dtype=float)

    return AggregateStats(
        best=float(arr_len.min()),
        mean=float(arr_len.mean()),
        std=float(arr_len.std(ddof=1)) if arr_len.size > 1 else 0.0,
        min_runtime=float(arr_time.min()),
        mean_runtime=float(arr_time.mean()),
        max_runtime=float(arr_time.max()),
    )


def compute_approximation_ratios(
    results: Iterable[RunResult],
    optimal_tour_length: Optional[float],
) -> Optional[np.ndarray]:
    if optimal_tour_length is None or optimal_tour_length <= 0.0:
        return None
    lens = np.array([r.tour_length for r in results], dtype=float)
    return lens / float(optimal_tour_length)
