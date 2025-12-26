"""
Simulated Annealing (SA) for the metric Traveling Salesperson Problem (TSP).

The state is a permutation of city indices. The neighbourhood move is a
2-opt reversal, which preserves a valid Hamiltonian cycle and tends to
produce locally shorter tours.

This implementation prioritises clarity and reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from utils.distance import tour_length


TwoOptMove = Tuple[int, int]


def random_initial_tour(n_cities: int, rng: np.random.Generator) -> List[int]:
    """Generate a random initial tour."""
    tour = list(range(n_cities))
    rng.shuffle(tour)
    return tour


def random_two_opt_move(n_cities: int, rng: np.random.Generator) -> TwoOptMove:
    """
    Sample a random 2-opt move (i, j) with 0 <= i < j < n_cities.
    """
    i = rng.integers(0, n_cities - 1)
    j = rng.integers(i + 1, n_cities)
    return i, j


def apply_two_opt(tour: List[int], move: TwoOptMove) -> None:
    """
    Apply a 2-opt reversal to the tour in place.
    """
    i, j = move
    tour[i : j + 1] = reversed(tour[i : j + 1])


@dataclass
class SAParameters:
    initial_temperature: float = 100.0
    cooling_rate: float = 0.99
    min_temperature: float = 1e-3
    iterations_per_temperature: int = 100
    max_iterations: int = 100000


@dataclass
class SAResult:
    best_tour: List[int]
    best_length: float
    history: List[float]


def simulated_annealing_tsp(
    dist_matrix: np.ndarray,
    params: SAParameters,
    seed: int,
) -> SAResult:
    """
    Run Simulated Annealing on a TSP instance.
    """
    rng = np.random.default_rng(seed)
    n_cities = dist_matrix.shape[0]

    current_tour = random_initial_tour(n_cities, rng)
    current_length = tour_length(current_tour, dist_matrix)

    best_tour = list(current_tour)
    best_length = current_length

    temperature = params.initial_temperature
    iterations = 0
    history: List[float] = [best_length]

    while temperature > params.min_temperature and iterations < params.max_iterations:
        for _ in range(params.iterations_per_temperature):
            iterations += 1
            if iterations >= params.max_iterations:
                break

            move = random_two_opt_move(n_cities, rng)
            candidate_tour = list(current_tour)
            apply_two_opt(candidate_tour, move)
            candidate_length = tour_length(candidate_tour, dist_matrix)

            delta = candidate_length - current_length

            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current_tour = candidate_tour
                current_length = candidate_length

                if current_length < best_length:
                    best_length = current_length
                    best_tour = list(current_tour)

            history.append(best_length)

        temperature *= params.cooling_rate

    return SAResult(best_tour=best_tour, best_length=best_length, history=history)


__all__ = ["SAParameters", "SAResult", "simulated_annealing_tsp"]
