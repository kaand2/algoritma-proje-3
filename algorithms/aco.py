"""
Ant Colony Optimization (ACO) for the metric Traveling Salesperson Problem (TSP).

We implement a standard Ant System variant:

- Each ant constructs a complete tour using a probabilistic transition rule
  that combines pheromone trails and heuristic desirability (inverse distance).
- After each iteration, pheromones evaporate globally and are reinforced
  based on the best tour found in that iteration.

The focus is on clarity and reproducibility rather than aggressive optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from utils.distance import tour_length


@dataclass
class ACOParameters:
    """
    Hyper-parameters for Ant Colony Optimization.

    Attributes
    ----------
    n_ants:
        Number of ants (usually comparable to the number of cities).
    alpha:
        Exponent for pheromone influence in the transition rule.
    beta:
        Exponent for heuristic influence (1 / distance).
    evaporation_rate:
        Pheromone evaporation factor rho in (0, 1).
    n_iterations:
        Number of iterations (generations of ants).
    initial_pheromone:
        Initial value for all pheromone trails.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 5.0
    evaporation_rate: float = 0.5
    n_iterations: int = 100
    initial_pheromone: float = 1.0


@dataclass
class ACOResult:
    """
    Result of running Ant Colony Optimization on one instance.

    Attributes
    ----------
    best_tour:
        Best tour (permutation of city indices) found during the run.
    best_length:
        Length of the best tour.
    history:
        Best tour length after each iteration.
    """

    best_tour: List[int]
    best_length: float
    history: List[float]


def _construct_tour_for_ant(
    start_city: int,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    params: ACOParameters,
    rng: np.random.Generator,
) -> List[int]:
    """
    Construct a tour for a single ant using the probabilistic transition rule.
    """
    n_cities = pheromone.shape[0]
    unvisited = list(range(n_cities))
    tour: List[int] = [start_city]
    unvisited.remove(start_city)

    current = start_city
    while unvisited:
        candidates = np.array(unvisited, dtype=int)
        tau = pheromone[current, candidates] ** params.alpha
        eta = heuristic[current, candidates] ** params.beta
        weights = tau * eta
        total = float(weights.sum())
        if total <= 0.0:
            # Fallback: choose uniformly among remaining cities.
            probabilities = np.ones_like(weights) / len(weights)
        else:
            probabilities = weights / total

        next_city = int(rng.choice(candidates, p=probabilities))
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return tour


def ant_colony_optimization_tsp(
    dist_matrix: np.ndarray,
    params: ACOParameters,
    seed: int,
) -> ACOResult:
    """
    Run Ant Colony Optimization on a TSP instance defined by its distance matrix.

    Parameters
    ----------
    dist_matrix:
        Symmetric distance matrix of shape (n_cities, n_cities).
    params:
        ACO hyper-parameters.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ACOResult
        Container with the best tour and its length.
    """
    rng = np.random.default_rng(seed)
    n_cities = dist_matrix.shape[0]

    # Heuristic desirability: inverse distance, with safe handling of zeros.
    heuristic = np.zeros_like(dist_matrix)
    with np.errstate(divide="ignore"):
        heuristic[dist_matrix > 0] = 1.0 / dist_matrix[dist_matrix > 0]

    pheromone = np.full_like(dist_matrix, fill_value=params.initial_pheromone, dtype=float)

    best_tour: List[int] = []
    best_length: float = float("inf")
    history: List[float] = []

    for _ in range(params.n_iterations):
        iteration_best_tour: List[int] | None = None
        iteration_best_length: float = float("inf")

        # Each ant constructs a tour.
        for _ant in range(params.n_ants):
            start_city = int(rng.integers(0, n_cities))
            tour = _construct_tour_for_ant(
                start_city=start_city,
                pheromone=pheromone,
                heuristic=heuristic,
                params=params,
                rng=rng,
            )
            length = tour_length(tour, dist_matrix)

            if length < iteration_best_length:
                iteration_best_length = length
                iteration_best_tour = tour

        rho = params.evaporation_rate
        pheromone *= (1.0 - rho)
        pheromone = np.maximum(pheromone, 1e-12)

        # Reinforcement by iteration-best ant.
        assert iteration_best_tour is not None
        deposit = 1.0 / iteration_best_length
        for i in range(n_cities):
            j = (i + 1) % n_cities
            a = iteration_best_tour[i]
            b = iteration_best_tour[j]
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit  # symmetric

        if iteration_best_length < best_length:
            best_length = iteration_best_length
            best_tour = list(iteration_best_tour)

        history.append(best_length)

    return ACOResult(best_tour=best_tour, best_length=best_length, history=history)


__all__ = ["ACOParameters", "ACOResult", "ant_colony_optimization_tsp"]


