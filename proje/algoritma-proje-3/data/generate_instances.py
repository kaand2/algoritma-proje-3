from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from utils.distance import euclidean_distance_matrix, tour_length


@dataclass
class TSPInstance:
    name: str
    coordinates: np.ndarray
    optimal_tour_length: float | None = None


def generate_random_coordinates(
    n_cities: int,
    low: float = 0.0,
    high: float = 100.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low=low, high=high, size=(n_cities, 2))


def brute_force_optimal_tour_length(coords: np.ndarray) -> float:
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
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{instance.name}.npz")
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
    rng = np.random.default_rng(random_seed)

    sizes_small: List[int] = [10, 11, 12]
    sizes_medium: List[int] = [50]
    sizes_large: List[int] = [100, 150, 200]

    instances: Dict[str, TSPInstance] = {}

    for n in sizes_small:
        coords = generate_random_coordinates(n, rng=rng)
        opt_len = brute_force_optimal_tour_length(coords)
        name = f"tsp_small_{n}"
        inst = TSPInstance(name=name, coordinates=coords, optimal_tour_length=opt_len)
        save_instance(inst, output_dir, random_seed=random_seed)
        instances[name] = inst

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
    data = np.load(path)
    coords = data["coordinates"]
    raw_opt = float(data["optimal_tour_length"])
    opt_len = None if raw_opt < 0.0 else raw_opt
    name = os.path.splitext(os.path.basename(path))[0]
    return TSPInstance(name=name, coordinates=coords, optimal_tour_length=opt_len)


def list_instance_files(output_dir: str = "data/instances") -> List[str]:
    if not os.path.isdir(output_dir):
        return []
    files: List[str] = []
    for fname in os.listdir(output_dir):
        if fname.endswith(".npz"):
            files.append(os.path.join(output_dir, fname))
    files.sort()
    return files


if __name__ == "__main__":
    instances = generate_instances()
    print(f"Generated {len(instances)} TSP instances in 'data/instances/'.")
