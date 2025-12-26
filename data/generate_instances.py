"""
Data generation utilities for metric TSP instances.

This module creates synthetic TSP instances with cities in the 2D plane
and (optionally) computes exact optimal tours for small instances
using brute-force search.

Generated instances are saved as NumPy ``.npz`` files so that algorithms
and experiments can load them in a reproducible way.

All randomness is controlled via an explicit random seed.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from utils.distance import euclidean_distance_matrix, tour_length


@dataclass
class TSPInstance:
    """
    Simple container for a TSP instance.

    Attributes
    ----------
    name:
        Human-readable identifier for the instance.
    coordinates:
        Array of shape (n_cities, 2) with (x, y) coordinates.
    optimal_tour_length:
        Exact optimal tour length if known, otherwise None.
    """

    name: str
    coordinates: np.ndarray
    optimal_tour_length: float | None = None


def generate_random_coordinates(
    n_cities: int,
    low: float = 0.0,
    high: float = 100.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate random 2D coordinates for TSP cities.

    Parameters
    ----------
    n_cities:
        Number of cities to generate.
    low, high:
        Range of coordinates (uniform in [low, high]).
    rng:
        Optional NumPy random Generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_cities, 2).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=low, high=high, size=(n_cities, 2))


def brute_force_optimal_tour_length(coords: np.ndarray) -> float:
    """
    Compute the exact optimal tour length via brute-force enumeration.

    This is only feasible for very small instances (e.g. n <= 12).
    The implementation fixes city 0 as the starting point and enumerates
    all permutations of the remaining cities to avoid equivalent tours
    due to rotation.

    Parameters
    ----------
    coords:
        Array of shape (n_cities, 2).

    Returns
    -------
    float
        Optimal tour length.
    """
    n = coords.shape[0]
    if n <= 1:
        return 0.0

    dist_matrix = euclidean_distance_matrix(coords)
    best = float("inf")

    cities = list(range(n))
    start = cities[0]
    others = cities[1:]

    for perm in itertools.permutations(others):
        tour = (start,) + perm
        length = tour_length(list(tour), dist_matrix)
        if length < best:
            best = length

    return best


def save_instance(
    instance: TSPInstance,
    output_dir: str,
    random_seed: int | None = None,
) -> str:
    """
    Save a TSP instance to disk as ``.npz``.

    The file includes coordinates and (optionally) the optimal tour length.

    Parameters
    ----------
    instance:
        The instance to save.
    output_dir:
        Directory where the file will be stored.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{instance.name}.npz")
    # Store coordinates, (optional) optimal tour length, and (optional) seed
    # used for generation. The seed is not strictly necessary for
    # reproducibility (the full data are stored), but it can be useful for
    # documentation and further studies.
    if random_seed is None:
        np.savez(
            path,
            coordinates=instance.coordinates,
            optimal_tour_length=(
                -1.0 if instance.optimal_tour_length is None else instance.optimal_tour_length
            ),
        )
    else:
        np.savez(
            path,
            coordinates=instance.coordinates,
            optimal_tour_length=(
                -1.0 if instance.optimal_tour_length is None else instance.optimal_tour_length
            ),
            random_seed=int(random_seed),
        )
    return path


def generate_instances(
    output_dir: str = "data/instances",
    random_seed: int = 42,
) -> Dict[str, TSPInstance]:
    """
    Generate a small set of benchmark TSP instances of different sizes.

    Instances
    ---------
    - Small:    10, 11, 12 cities (with exact optimum by brute force)
    - Medium:   50 cities
    - Large:    100, 150, 200 cities

    Parameters
    ----------
    output_dir:
        Directory where instance files will be written.
    random_seed:
        Seed for the RNG to ensure reproducibility.

    Returns
    -------
    Dict[str, TSPInstance]
        Mapping from instance name to instance object.
    """
    rng = np.random.default_rng(random_seed)

    sizes_small: List[int] = [10, 11, 12]
    sizes_medium: List[int] = [50]
    sizes_large: List[int] = [100, 150, 200]

    instances: Dict[str, TSPInstance] = {}

    # Small instances with optimal tour length.
    for n in sizes_small:
        coords = generate_random_coordinates(n, rng=rng)
        opt_len = brute_force_optimal_tour_length(coords)
        name = f"tsp_small_{n}"
        inst = TSPInstance(name=name, coordinates=coords, optimal_tour_length=opt_len)
        save_instance(inst, output_dir, random_seed=random_seed)
        instances[name] = inst

    # Medium and large instances without optimal tour length.
    for n in sizes_medium:
        coords = generate_random_coordinates(n, rng=rng)
        name = f"tsp_medium_{n}"
        inst = TSPInstance(name=name, coordinates=coords, optimal_tour_length=None)
        save_instance(inst, output_dir, random_seed=random_seed)
        instances[name] = inst

    for n in sizes_large:
        coords = generate_random_coordinates(n, rng=rng)
        name = f"tsp_large_{n}"
        inst = TSPInstance(name=name, coordinates=coords, optimal_tour_length=None)
        save_instance(inst, output_dir, random_seed=random_seed)
        instances[name] = inst

    return instances


def load_instance(path: str) -> TSPInstance:
    """
    Load a TSP instance from a ``.npz`` file created by :func:`save_instance`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.

    Returns
    -------
    TSPInstance
        Loaded instance.
    """
    data = np.load(path)
    coords = data["coordinates"]
    raw_opt = float(data["optimal_tour_length"])
    opt_len = None if raw_opt < 0.0 else raw_opt
    name = os.path.splitext(os.path.basename(path))[0]
    return TSPInstance(name=name, coordinates=coords, optimal_tour_length=opt_len)


def list_instance_files(output_dir: str = "data/instances") -> List[str]:
    """
    List all available instance files in the given directory.

    Parameters
    ----------
    output_dir:
        Directory to search for ``.npz`` files.

    Returns
    -------
    List[str]
        Sorted list of file paths.
    """
    if not os.path.isdir(output_dir):
        return []
    files: List[str] = []
    for fname in os.listdir(output_dir):
        if fname.endswith(".npz"):
            files.append(os.path.join(output_dir, fname))
    files.sort()
    return files


if __name__ == "__main__":
    # Simple CLI entry point to regenerate all benchmark instances.
    instances = generate_instances()
    print(f"Generated {len(instances)} TSP instances in 'data/instances/'.")


