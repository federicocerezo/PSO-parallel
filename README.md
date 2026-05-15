# PSO-parallel

Particle Swarm Optimization (PSO) in Python. Compares sequential, threading, multiprocessing, asyncio and NumPy evaluation strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Project structure

```
core/           PSO engine: particle, swarm, pso loop + abstractions (evaluator, bounds, topology)
objectives/     Benchmark functions: sphere, ackley, rosenbrock, rastrigin, noisy_sphere
parallel/       Evaluator implementations: sequential (V0), threading (V1), multiprocessing (V2), asyncio (V3)
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
python3 scripts/run_pso.py --objective sphere --dim 10 --seed 42
python3 scripts/run_pso.py --objective rastrigin --dim 30 --evaluator threading
python3 scripts/run_pso.py --objective ackley --dim 10 --evaluator multiprocessing
python3 scripts/run_pso.py --objective noisy_sphere --dim 10 --evaluator asyncio
python3 scripts/run_pso.py --objective sphere --dim 10 --evaluator numpy
python3 scripts/run_pso.py --objective sphere --dim 10 --compare-baseline
python3 scripts/run_pso.py --help
```

### Benchmark suite (all objectives × dimensions × seeds)

```bash
python3 scripts/run_benchmarks.py
python3 scripts/run_benchmarks.py --evaluator threading
python3 scripts/run_benchmarks.py --evaluator multiprocessing --max-iters 1000
python3 scripts/run_benchmarks.py --evaluator asyncio
```

### Hyperparameter grid search

```bash
python3 scripts/run_grid_search.py --objective sphere --dim 10
python3 scripts/run_grid_search.py --objective rastrigin --dim 30 --max-iters 500
```

### Visualization

```bash
python3 scripts/make_viz.py --objective sphere                        # convergence plot + GIF animation (d=2)
python3 scripts/make_viz.py --objective ackley --dim 2 --max-iters 80
python3 scripts/make_viz.py --objective rastrigin --dim 2 --fps 15
```

### Analysis (compare strategies, generate tables and plots)

```bash
python3 scripts/run_analysis.py
python3 scripts/run_analysis.py --results-dir results --save-dir results/analysis
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
| V3 | `parallel/asyncio_eval.py` | asyncio.gather + run_in_executor | Done |
| V4 | `parallel/numpy_eval.py` | NumPy batch operations (vectorized) | Done |

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

### V3 — asyncio (cooperative concurrency)

Uses `asyncio.gather` to launch all particle evaluations concurrently. Each
coroutine wraps the sync objective with `loop.run_in_executor`, which dispatches
it to the default thread pool. While one thread is blocked on I/O, the event
loop schedules the others — total wall time ≈ max(latencies) instead of sum(latencies).

**When it makes sense**: only for I/O-bound objectives. The benchmark function
`noisy_sphere` (`objectives/noisy_service.py`) simulates this by adding a random
delay of 5–50 ms per evaluation (modelling a query to a local service or simulator).
On CPU-bound functions like Sphere or Rastrigin, V3 provides no advantage over V1
since threads still compete for the GIL.

```bash
# Recommended use: pair asyncio with the noisy_sphere objective
python3 scripts/run_pso.py --objective noisy_sphere --dim 10 --evaluator asyncio
```

### V4 — NumPy vectorized (implicit parallelism)

Stacks all particle positions into a single `(n_particles, dim)` matrix and
evaluates them in one NumPy call using broadcasting. Avoids the Python loop
over particles entirely. The speedup comes from NumPy's internal C/BLAS
routines, not from threads or processes.

Vectorized implementations for all four standard benchmarks are in
`objectives/vectorized.py`. `noisy_sphere` has no batch version (its
`time.sleep` cannot be vectorized).

```bash
python3 scripts/run_pso.py --objective sphere --dim 30 --evaluator numpy
python3 scripts/run_benchmarks.py --evaluator numpy
```

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
