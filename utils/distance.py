"""
Distance and tour-length utilities for the TSP.

This module centralises all distance computations to guarantee that
Simulated Annealing, ACO, and evaluation use exactly the same metric.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix.

    Parameters
    ----------
    coords:
        Array of shape (n_cities, 2) with (x, y) coordinates.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_cities, n_cities) where entry (i, j) is the
        Euclidean distance between city i and city j.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    return dist_matrix


def tour_length(tour: Iterable[int], dist_matrix: np.ndarray) -> float:
    """
    Compute the total length of a Hamiltonian cycle.

    The tour is assumed to be a permutation of all city indices.
    The cycle is closed by returning from the last city back to the first.

    Parameters
    ----------
    tour:
        Sequence of city indices, e.g. [0, 2, 1, 3].
    dist_matrix:
        Precomputed pairwise distance matrix.

    Returns
    -------
    float
        Total tour length.
    """
    indices: List[int] = list(tour)
    if len(indices) == 0:
        return 0.0
    total = 0.0
    n = len(indices)
    for i in range(n):
        j = (i + 1) % n
        a = indices[i]
        b = indices[j]
        total += float(dist_matrix[a, b])
    return total


