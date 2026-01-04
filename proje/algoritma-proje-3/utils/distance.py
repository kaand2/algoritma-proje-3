from __future__ import annotations

from typing import Iterable, List

import numpy as np


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    return dist_matrix


def tour_length(tour: Iterable[int], dist_matrix: np.ndarray) -> float:
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
