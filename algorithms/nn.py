"""
Nearest Neighbor (NN) heuristic for the metric Traveling Salesperson Problem (TSP).

A simple greedy baseline that:
1. Starts from a city (optionally randomized)
2. Repeatedly visits the closest unvisited city
3. Returns to the start at the end

This deterministic heuristic provides a baseline to compare against more
advanced metaheuristics like Simulated Annealing and Ant Colony Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.distance import tour_length


@dataclass
class NNParameters:
    """
    Hyper-parameters for Nearest Neighbor heuristic.

    Attributes
    ----------
    randomize_start:
        If True, use the seed to randomly select the starting city.
        If False, always start from city 0.
    """

    randomize_start: bool = True


@dataclass
class NNResult:
    """
    Result of running Nearest Neighbor on one instance.

    Attributes
    ----------
    best_tour:
        The tour (permutation of city indices) found by NN.
    best_length:
        Length of the tour.
    history:
        History of best tour length (single value for NN since it's deterministic).
    """

    best_tour: List[int]
    best_length: float
    history: List[float]


def nearest_neighbor_tsp(
    dist_matrix: np.ndarray,
    params: NNParameters,
    seed: int,
) -> NNResult:
    """
    Run Nearest Neighbor heuristic on a TSP instance.

    Parameters
    ----------
    dist_matrix:
        Symmetric distance matrix of shape (n_cities, n_cities).
    params:
        NN hyper-parameters.
    seed:
        Random seed for reproducibility (used only if randomize_start=True).

    Returns
    -------
    NNResult
        Container with the tour and its length.
    """
    n = dist_matrix.shape[0]
    rng = np.random.default_rng(seed)

    # Select starting city
    if params.randomize_start:
        start = int(rng.integers(0, n))
    else:
        start = 0

    # Build tour using nearest neighbor
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)

    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    # Compute tour length
    best_length = tour_length(tour, dist_matrix)

    return NNResult(
        best_tour=tour,
        best_length=best_length,
        history=[best_length],
    )


__all__ = ["NNParameters", "NNResult", "nearest_neighbor_tsp"]

