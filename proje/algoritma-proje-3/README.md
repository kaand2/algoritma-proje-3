# Traveling Salesman Problem (TSP) Solver

A Python implementation comparing three heuristic algorithms for solving the Traveling Salesman Problem: Ant Colony Optimization (ACO), Simulated Annealing (SA), and Nearest Neighbor (NN).

## Overview

This project implements and evaluates three different approaches to solve the TSP:

- **Nearest Neighbor (NN)**: A greedy constructive heuristic that builds a tour by always visiting the nearest unvisited city.
- **Simulated Annealing (SA)**: A metaheuristic that uses a cooling schedule to escape local optima, employing 2-opt moves for neighborhood exploration.
- **Ant Colony Optimization (ACO)**: A swarm intelligence algorithm that uses pheromone trails and heuristic information to construct tours.

## Project Structure

```
algoritma-proje-3/
├── algorithms/          # Algorithm implementations
│   ├── aco.py          # Ant Colony Optimization
│   ├── sa.py           # Simulated Annealing
│   └── nn.py           # Nearest Neighbor
├── data/               # TSP instance generation and storage
│   ├── generate_instances.py
│   ├── instances/      # Generated TSP instances
│   └── instances_test/ # Test instances
├── experiments/        # Experiment runner
│   └── run_experiments.py
├── utils/             # Utility functions
│   ├── distance.py    # Distance matrix and tour length calculations
│   └── evaluation.py  # Evaluation metrics and statistics
└── report_assets/     # Generated reports (plots and tables)
```

## Features

- **Multiple Algorithms**: Compare three different TSP solving approaches
- **Instance Generation**: Generate random TSP instances of various sizes
- **Comprehensive Evaluation**: Track tour lengths, runtimes, and approximation ratios
- **Visualization**: Generate plots comparing algorithm performance
- **Reproducibility**: Seed-based random number generation for consistent results

## Dependencies

- `numpy >= 1.20.0`
- `matplotlib >= 3.3.0`

## Usage

### Running Experiments

To run all experiments and generate results:

```bash
python experiments/run_experiments.py
```

This will:
1. Generate or load TSP instances from `data/instances/`
2. Run each algorithm on each instance with multiple random seeds
3. Generate a CSV file with detailed results in `report_assets/tables.csv`
4. Create visualization plots in `report_assets/plots.png`

### Using Individual Algorithms

#### Nearest Neighbor

```python
from algorithms.nn import nearest_neighbor_tsp, NNParameters
from utils.distance import euclidean_distance_matrix
import numpy as np

# Create distance matrix from coordinates
coords = np.array([[0, 0], [1, 1], [2, 0], [1, -1]])
dist_matrix = euclidean_distance_matrix(coords)

# Run algorithm
params = NNParameters(randomize_start=True)
result = nearest_neighbor_tsp(dist_matrix, params, seed=42)

print(f"Tour: {result.best_tour}")
print(f"Length: {result.best_length}")
```

#### Simulated Annealing

```python
from algorithms.sa import simulated_annealing_tsp, SAParameters

params = SAParameters(
    initial_temperature=100.0,
    cooling_rate=0.98,
    min_temperature=1e-3,
    iterations_per_temperature=200,
    max_iterations=50000
)
result = simulated_annealing_tsp(dist_matrix, params, seed=42)
```

#### Ant Colony Optimization

```python
from algorithms.aco import ant_colony_optimization_tsp, ACOParameters

params = ACOParameters(
    n_ants=20,
    alpha=1.0,
    beta=5.0,
    evaporation_rate=0.5,
    n_iterations=100,
    initial_pheromone=1.0
)
result = ant_colony_optimization_tsp(dist_matrix, params, seed=42)
```

## Algorithm Parameters

### Nearest Neighbor (NN)
- `randomize_start`: Whether to start from a random city (default: `True`)

### Simulated Annealing (SA)
- `initial_temperature`: Starting temperature (default: `100.0`)
- `cooling_rate`: Temperature reduction factor per iteration (default: `0.99`)
- `min_temperature`: Minimum temperature threshold (default: `1e-3`)
- `iterations_per_temperature`: Number of iterations at each temperature (default: `100`)
- `max_iterations`: Maximum total iterations (default: `100000`)

### Ant Colony Optimization (ACO)
- `n_ants`: Number of ants in the colony (default: `20`)
- `alpha`: Pheromone importance parameter (default: `1.0`)
- `beta`: Heuristic importance parameter (default: `5.0`)
- `evaporation_rate`: Pheromone evaporation rate (default: `0.5`)
- `n_iterations`: Number of iterations (default: `100`)
- `initial_pheromone`: Initial pheromone value (default: `1.0`)

## Output

The experiment runner generates:

1. **`report_assets/tables.csv`**: Detailed results for each run including:
   - Instance name and number of cities
   - Algorithm used
   - Random seed
   - Tour length and runtime
   - Optimal tour length (if available)
   - Approximation ratio

2. **`report_assets/plots.png`**: Visualization plots showing:
   - Tour length vs problem size
   - Runtime vs problem size
   - Approximation ratio comparison

## Instance Sizes

The project includes TSP instances of various sizes:
- Small: 10-12 cities
- Medium: 50 cities
- Large: 100, 150, 200 cities

## License

This project is part of an algorithms course assignment.

