# PSO-parallel

Particle Swarm Optimization (PSO) in Python with sequential, threading, multiprocessing, asyncio and NumPy evaluation strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Project structure

```
core/           PSO engine: particle, swarm, pso loop, evaluator/bounds/topology abstractions
objectives/     Benchmark functions: sphere, ackley, rosenbrock, rastrigin
parallel/       Concurrent and parallel evaluators (V1–V4, added progressively)
experiments/    Orchestration: runner, grid search
storage/        Persistence (JSON/CSV) and logging
viz/            Convergence plots and swarm animations
tests/          Unit tests
results/        Experiment outputs (gitignored)
logs/           Run logs (gitignored)
```

## Commands

### Single run

```bash
python3 run_pso.py --objective sphere --dim 10 --seed 42
python3 run_pso.py --objective rastrigin --dim 30 --n-particles 50 --max-iters 1000
python3 run_pso.py --help
```

### Benchmark suite (all objectives × dimensions × seeds)

```bash
python3 run_benchmarks.py
python3 run_benchmarks.py --max-iters 1000 --n-particles 50
```

### Hyperparameter grid search

```bash
python3 run_grid_search.py --objective sphere --dim 10
python3 run_grid_search.py --objective ackley --dim 30 --max-iters 500
```

### Visualization (d=2 animation + convergence plot)

```bash
python3 make_viz.py --objective sphere
python3 make_viz.py --objective rastrigin --n-particles 30 --max-iters 80
```

### Tests

```bash
python3 -m pytest tests/ -v
```

## Parallelism strategies

| Version | Strategy | Status |
|---------|----------|--------|
| V0 | Sequential (baseline) | Done |
| V1 | threading (ThreadPoolExecutor) | Pending |
| V2 | multiprocessing (ProcessPoolExecutor) | Pending |
| V3 | asyncio (cooperative, simulated latency) | Pending |
| V4 | NumPy vectorized | Pending |

## Bounds strategy

ClampBounds is used: positions are clipped to [lo, hi] and velocity is zeroed on collision axes. This avoids oscillation near walls.

## Reproducibility

All runs accept `--seed`. The seed is recorded in the JSON summary alongside git hash and hardware info.

## Results format

Each run writes to `results/<objective>_d<dim>_s<seed>/`:
- `summary.json` — full config, metrics, git hash, hardware
- `history.csv` — best fitness per iteration
