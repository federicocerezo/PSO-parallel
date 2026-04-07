# PSO-parallel

Particle Swarm Optimization (PSO) in Python. Compares sequential, threading, multiprocessing, asyncio and NumPy evaluation strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Project structure

```
core/           PSO engine: particle, swarm, pso loop + abstractions (evaluator, bounds, topology)
objectives/     Benchmark functions: sphere, ackley, rosenbrock, rastrigin
parallel/       Evaluator implementations: sequential (V0), threading (V1), multiprocessing (V2)
experiments/    Orchestration: runner, grid search
storage/        Persistence (JSON/CSV) and structured logging
viz/            Convergence plots
tests/          Unit tests
results/        Experiment outputs (gitignored)
logs/           Run logs (gitignored)
```

## Commands

### Single run

```bash
python3 run_pso.py --objective sphere --dim 10 --seed 42
python3 run_pso.py --objective rastrigin --dim 30 --evaluator threading
python3 run_pso.py --objective ackley --dim 10 --evaluator multiprocessing
python3 run_pso.py --objective sphere --dim 10 --compare-baseline
python3 run_pso.py --help
```

### Benchmark suite (all objectives × dimensions × seeds)

```bash
python3 run_benchmarks.py
python3 run_benchmarks.py --evaluator threading
python3 run_benchmarks.py --evaluator multiprocessing --max-iters 1000
```

### Hyperparameter grid search

```bash
python3 run_grid_search.py --objective sphere --dim 10
python3 run_grid_search.py --objective rastrigin --dim 30 --max-iters 500
```

### Visualization

```bash
python3 make_viz.py --objective sphere                        # convergence plot + GIF animation (d=2)
python3 make_viz.py --objective ackley --dim 2 --max-iters 80
python3 make_viz.py --objective rastrigin --dim 2 --fps 15
```

### Analysis (compare strategies, generate tables and plots)

```bash
python3 run_analysis.py
python3 run_analysis.py --results-dir results --save-dir results/analysis
```

### Tests

```bash
python3 -m pytest tests/ -v
```

## Design document

See [docs/design.md](docs/design.md) for architecture decisions, trade-offs, and limitations.

## Parallelism strategies

| Version | File | Strategy | Status |
|---------|------|----------|--------|
| V0 | `parallel/sequential.py` | Sequential loop | Done |
| V1 | `parallel/threading_eval.py` | ThreadPoolExecutor | Done |
| V2 | `parallel/multiprocessing_eval.py` | ProcessPoolExecutor + batching | Done |
| V3 | — | asyncio (simulated latency) | Pending |
| V4 | — | NumPy vectorized | Pending |

All versions share the same PSO core and produce identical results for the same seed.
Only the fitness evaluation strategy changes.

### V0 — Sequential

Evaluates each particle one by one. Baseline reference.

### V1 — Threading (ThreadPoolExecutor)

Distributes fitness evaluation across threads. Python's GIL prevents true parallelism
for CPU-bound code, so V1 is typically slower than V0 for fast benchmark functions.
It would benefit from objectives that release the GIL (NumPy-heavy or I/O-bound work).

### V2 — Multiprocessing (ProcessPoolExecutor)

Each worker runs in a separate process with its own interpreter — no GIL.
Positions are serialized (pickled) and sent to worker processes, results are
deserialized back. This IPC overhead dominates for fast functions like Sphere,
making V2 slower than V0. For expensive objectives (>10ms per evaluation),
V2 provides genuine speedup.

Batching is used to reduce the number of IPC round-trips: particles are grouped
into chunks and each chunk is sent to a worker as a single task.

## Bounds strategy

`ClampBounds`: positions are clipped to `[lo, hi]` and velocity is zeroed on
collision axes. This prevents oscillation near walls and is simple to reason about.

## Reproducibility

All runs accept `--seed`. The seed controls `np.random.default_rng(seed)` which
initializes the swarm and all random perturbations. The seed is recorded in the
JSON summary alongside git hash and hardware info.

## Results format

Each run saves to `results/<objective>_d<dim>_s<seed>/`:
- `summary.json` — full config, metrics, git hash, hardware
- `history.csv` — best fitness per iteration
