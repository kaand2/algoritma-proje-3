"""
Evaluation utilities for TSP algorithms.

Provides helpers to compute basic statistics over multiple runs
and approximation ratios where the optimal tour length is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RunResult:
    """
    Container for the result of a single algorithm run.

    Attributes
    ----------
    tour_length:
        Total length of the best tour found.
    runtime_sec:
        Wall-clock runtime in seconds.
    seed:
        Random seed used for the run.
    """

    tour_length: float
    runtime_sec: float
    seed: int


@dataclass
class AggregateStats:
    """
    Aggregate statistics across multiple runs on the same instance.

    Attributes
    ----------
    best:
        Best (minimum) tour length observed.
    mean:
        Mean tour length.
    std:
        Standard deviation of tour lengths.
    min_runtime:
        Fastest runtime.
    mean_runtime:
        Mean runtime.
    max_runtime:
        Slowest runtime.
    """

    best: float
    mean: float
    std: float
    min_runtime: float
    mean_runtime: float
    max_runtime: float


def compute_aggregate_stats(results: Iterable[RunResult]) -> AggregateStats:
    """
    Compute basic aggregate statistics for a collection of runs.
    """
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
    """
    Compute approximation ratios tour_length / optimal_tour_length
    for a set of runs.

    Parameters
    ----------
    results:
        Collection of run results.
    optimal_tour_length:
        Exact optimum. If None or non-positive, returns None.

    Returns
    -------
    np.ndarray or None
        Array of ratios if optimal_tour_length is valid, otherwise None.
    """
    if optimal_tour_length is None or optimal_tour_length <= 0.0:
        return None
    lens = np.array([r.tour_length for r in results], dtype=float)
    return lens / float(optimal_tour_length)


