## Traveling Salesperson Problem Metaheuristics Project

This project implements and experimentally evaluates two metaheuristics for
the metric Traveling Salesperson Problem (TSP):

- **Simulated Annealing (SA)**
- **Ant Colony Optimization (ACO)**

Cities are points in the 2D plane, distances are Euclidean, and instances
are generated synthetically.

The code aims to be suitable for MSc-level coursework: it is fully
reproducible, parameterised, and designed to support a written report.

---

## Project structure

- **`data/generate_instances.py`**: Generate synthetic TSP instances
  (small/medium/large) and exact optima for small instances.
- **`algorithms/sa.py`**: Simulated Annealing with 2-opt neighbourhood.
- **`algorithms/aco.py`**: Ant Colony Optimization with pheromone trails
  and heuristic information.
- **`utils/distance.py`**: Distance matrix and tour-length utilities.
- **`utils/evaluation.py`**: Run-level containers and simple statistics.
- **`experiments/run_experiments.py`**: Experiment driver that runs SA
  and ACO, saves CSV tables and plots.
- **`report_assets/`**: Output directory for tables and figures.

---

## Setup

It is recommended to use the provided virtual environment or a fresh one.

1. Create and activate a virtual environment (if needed):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows PowerShell
```

2. Install required Python packages:

```bash
pip install numpy matplotlib
```

---

## How to run experiments

From the project root:

```bash
venv\Scripts\activate  # if not already active
python -m experiments.run_experiments
```

What this does:

- Ensures TSP instances exist in `data/instances/` (generates them if missing).
- Runs **SA** and **ACO** on all instances (small/medium/large).
- Uses multiple random seeds for each (algorithm, instance) pair.
- Writes a single long-form CSV table to:
  - `report_assets/tables.csv`
- Generates plots and saves them to:
  - `report_assets/plots.png`

These artefacts are ready to be used directly in a report.

---

## Algorithm parameters

### Simulated Annealing (SA)

Defined in `algorithms/sa.py` via `SAParameters`:

- **`initial_temperature`**: Starting temperature \\(T_0\\).
- **`cooling_rate`**: Geometric cooling factor \\(\\alpha\\) in (0, 1);
  temperature is updated as \\(T \\leftarrow \\alpha T\\).
- **`min_temperature`**: Stop when temperature drops below this value.
- **`iterations_per_temperature`**: Neighbour evaluations at each temperature.
- **`max_iterations`**: Global cap on the total number of iterations.

State: permutation of cities.  
Neighbourhood: 2-opt reversal.  
Acceptance: always accept improvements, accept worse moves with probability
\\(\\exp(-\\Delta E / T)\\).

### Ant Colony Optimization (ACO)

Defined in `algorithms/aco.py` via `ACOParameters`:

- **`n_ants`**: Number of ants constructing tours each iteration.
- **`alpha`**: Exponent for pheromone influence.
- **`beta`**: Exponent for heuristic influence (\\(\\eta = 1 / d\\)).
- **`evaporation_rate`**: Pheromone evaporation factor \\(\\rho\\).
- **`n_iterations`**: Number of iterations (generations).
- **`initial_pheromone`**: Initial value for all pheromone trails.

Transition rule: ants choose the next city proportionally to
\\(\\tau_{ij}^{\\alpha} \\eta_{ij}^{\\beta}\\), where
\\(\\tau_{ij}\\) is pheromone and \\(\\eta_{ij}\\) is heuristic desirability.
After each iteration, pheromones evaporate and the best ant of that
iteration reinforces its tour.

---

## Experiments and outputs

The default experiment configuration (in `experiments/run_experiments.py`) is:

- **Seeds per configuration**: 5 (0–4 inclusive).
- **SA**: moderately aggressive cooling, with a safety cap on iterations.
- **ACO**: typical parameter choices for small benchmark instances.

For each `(algorithm, instance)` pair we record, per seed:

- **Best tour length**.
- **Runtime** (seconds).
- **Approximation ratio** (small instances only), defined as:

\\[
\\text{ratio} = \\frac{\\text{algorithm\_solution}}{\\text{optimal\_solution}}
\\]

The CSV `report_assets/tables.csv` contains all runs in a long format
with columns:

- `instance_name`, `n_cities`, `algorithm`, `seed`,
- `tour_length`, `runtime_sec`,
- `optimal_tour_length`, `approx_ratio`.

The figure `report_assets/plots.png` contains three subplots:

- Tour length vs number of cities.
- Runtime vs number of cities.
- Approximation ratio vs number of cities (small instances only).

---

## Reproducibility notes

- All stochastic components are controlled via explicit seeds
  (`numpy.random.default_rng(seed)`).
- Instance generation is deterministic given the top-level seed.
- Experiments use a fixed list of seeds; changing this list allows for
  more extensive studies while preserving the current setup.


