from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from algorithms.aco import ACOParameters, ant_colony_optimization_tsp
from algorithms.nn import NNParameters, nearest_neighbor_tsp
from algorithms.sa import SAParameters, simulated_annealing_tsp
from data.generate_instances import (
    TSPInstance,
    generate_instances,
    list_instance_files,
    load_instance,
)
from utils.distance import euclidean_distance_matrix
from utils.evaluation import RunResult, compute_approximation_ratios


def ensure_instances() -> Dict[str, TSPInstance]:
    instance_dir = "data/instances"
    files = list_instance_files(instance_dir)
    if not files:
        return generate_instances(output_dir=instance_dir, random_seed=42)

    instances: Dict[str, TSPInstance] = {}
    for path in files:
        inst = load_instance(path)
        instances[inst.name] = inst
    return instances


def run_single_experiment(
    instance: TSPInstance,
    algorithm: str,
    seed: int,
    sa_params: SAParameters,
    aco_params: ACOParameters,
    nn_params: NNParameters,
) -> RunResult:
    coords = instance.coordinates
    dist = euclidean_distance_matrix(coords)

    start = time.perf_counter()
    if algorithm == "SA":
        res = simulated_annealing_tsp(dist, sa_params, seed)
        best_len = res.best_length
    elif algorithm == "ACO":
        res = ant_colony_optimization_tsp(dist, aco_params, seed)
        best_len = res.best_length
    elif algorithm == "NN":
        res = nearest_neighbor_tsp(dist, nn_params, seed)
        best_len = res.best_length
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    end = time.perf_counter()

    return RunResult(tour_length=best_len, runtime_sec=end - start, seed=seed)


def run_all_experiments() -> None:
    instances = ensure_instances()
    sorted_instances: List[Tuple[str, TSPInstance]] = sorted(
        instances.items(), key=lambda kv: (kv[1].coordinates.shape[0], kv[0])
    )

    seeds: List[int] = [0, 1, 2, 3, 4]

    sa_params = SAParameters(
        initial_temperature=100.0,
        cooling_rate=0.98,
        min_temperature=1e-3,
        iterations_per_temperature=200,
        max_iterations=50000,
    )
    aco_params = ACOParameters(
        n_ants=20,
        alpha=1.0,
        beta=5.0,
        evaporation_rate=0.5,
        n_iterations=100,
        initial_pheromone=1.0,
    )
    nn_params = NNParameters(randomize_start=True)

    os.makedirs("report_assets", exist_ok=True)
    csv_path = os.path.join("report_assets", "tables.csv")

    fieldnames = [
        "instance_name",
        "n_cities",
        "algorithm",
        "seed",
        "tour_length",
        "runtime_sec",
        "optimal_tour_length",
        "approx_ratio",
    ]

    rows: List[Dict[str, float]] = []

    for name, inst in sorted_instances:
        n_cities = inst.coordinates.shape[0]
        for algorithm in ("SA", "ACO", "NN"):
            per_run_results: List[RunResult] = []

            for seed in seeds:
                result = run_single_experiment(
                    instance=inst,
                    algorithm=algorithm,
                    seed=seed,
                    sa_params=sa_params,
                    aco_params=aco_params,
                    nn_params=nn_params,
                )
                per_run_results.append(result)

            ratios = compute_approximation_ratios(
                per_run_results, inst.optimal_tour_length
            )
            for idx, r in enumerate(per_run_results):
                ratio_val = float(ratios[idx]) if ratios is not None else float("nan")
                rows.append(
                    {
                        "instance_name": name,
                        "n_cities": n_cities,
                        "algorithm": algorithm,
                        "seed": r.seed,
                        "tour_length": r.tour_length,
                        "runtime_sec": r.runtime_sec,
                        "optimal_tour_length": inst.optimal_tour_length
                        if inst.optimal_tour_length is not None
                        else float("nan"),
                        "approx_ratio": ratio_val,
                    }
                )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    create_plots_from_rows(rows, output_path=os.path.join("report_assets", "plots.png"))


def create_plots_from_rows(
    rows: List[Dict[str, float]],
    output_path: str,
) -> None:
    by_inst_alg: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["instance_name"], row["algorithm"])
        by_inst_alg.setdefault(key, []).append(row)

    summary: List[Dict[str, float]] = []
    for (inst_name, alg), rlist in by_inst_alg.items():
        n_cities = rlist[0]["n_cities"]
        lengths = np.array([r["tour_length"] for r in rlist], dtype=float)
        runtimes = np.array([r["runtime_sec"] for r in rlist], dtype=float)
        approx_vals = np.array([r["approx_ratio"] for r in rlist], dtype=float)

        approx_mean = float("nan")
        approx_min = float("nan")
        approx_max = float("nan")
        if np.isfinite(approx_vals).any():
            finite_vals = approx_vals[np.isfinite(approx_vals)]
            approx_mean = float(np.nanmean(approx_vals))
            approx_min = float(np.nanmin(approx_vals))
            approx_max = float(np.nanmax(approx_vals))

        summary.append(
            {
                "instance_name": inst_name,
                "n_cities": n_cities,
                "algorithm": alg,
                "mean_length": float(lengths.mean()),
                "mean_runtime": float(runtimes.mean()),
                "mean_approx_ratio": approx_mean,
                "min_approx_ratio": approx_min,
                "max_approx_ratio": approx_max,
            }
        )

    algs = sorted({s["algorithm"] for s in summary})
    sizes = sorted({s["n_cities"] for s in summary})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for alg in algs:
        xs = []
        ys = []
        for n in sizes:
            vals = [
                s
                for s in summary
                if s["algorithm"] == alg and s["n_cities"] == n
            ]
            if not vals:
                continue
            xs.append(n)
            ys.append(np.mean([v["mean_length"] for v in vals]))
        if xs:
            axes[0].plot(xs, ys, marker="o", label=alg)
    axes[0].set_xlabel("Number of cities")
    axes[0].set_ylabel("Mean tour length")
    axes[0].set_title("Tour length vs problem size")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    for alg in algs:
        xs = []
        ys = []
        for n in sizes:
            vals = [
                s
                for s in summary
                if s["algorithm"] == alg and s["n_cities"] == n
            ]
            if not vals:
                continue
            xs.append(n)
            ys.append(np.mean([v["mean_runtime"] for v in vals]))
        if xs:
            axes[1].plot(xs, ys, marker="o", label=alg)
    axes[1].set_xlabel("Number of cities")
    axes[1].set_ylabel("Mean runtime (s)")
    axes[1].set_title("Runtime vs problem size")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    for alg in algs:
        xs = []
        ys = []
        yerr_lower = []
        yerr_upper = []
        for s in summary:
            if s["algorithm"] != alg:
                continue
            if not np.isfinite(s["mean_approx_ratio"]):
                continue
            xs.append(s["n_cities"])
            mean_val = s["mean_approx_ratio"]
            min_val = s["min_approx_ratio"]
            max_val = s["max_approx_ratio"]
            ys.append(mean_val)
            yerr_lower.append(max(0, mean_val - min_val) if np.isfinite(min_val) else 0)
            yerr_upper.append(max(0, max_val - mean_val) if np.isfinite(max_val) else 0)
        if xs:
            axes[2].errorbar(
                xs,
                ys,
                yerr=[yerr_lower, yerr_upper],
                marker="o",
                label=alg,
                capsize=4,
                capthick=1.5,
            )
    axes[2].set_xlabel("Number of cities (small instances)")
    axes[2].set_ylabel("Approximation ratio (min / mean / max)")
    axes[2].set_title("Approximation ratio (algorithm / optimum)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_all_experiments()
