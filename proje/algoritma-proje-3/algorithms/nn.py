from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.distance import tour_length


@dataclass
class NNParameters:
    randomize_start: bool = True


@dataclass
class NNResult:
    best_tour: List[int]
    best_length: float
    history: List[float]


def nearest_neighbor_tsp(
    dist_matrix: np.ndarray,
    params: NNParameters,
    seed: int,
) -> NNResult:
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(seed)

    if params.randomize_start:
        start = int(rng.integers(0, n))
    else:
        start = 0

    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)

    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    best_length = tour_length(tour, dist_matrix)

    return NNResult(
        best_tour=tour,
        best_length=best_length,
        history=[best_length],
    )


__all__ = ["NNParameters", "NNResult", "nearest_neighbor_tsp"]
