from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np
from numpy.typing import NDArray

from .swarm import build_swarm
from .evaluator import FitnessEvaluator
from .bounds import BoundsPolicy, ClampBounds
from .topology import Topology, GlobalBestTopology


@dataclass
class RunResult:
    best_position: NDArray[np.float64]
    best_fitness: float
    iterations: int
    time_total: float
    time_eval: float
    time_update: float
    history: List[float]


class PSO:
    def __init__(
        self,
        evaluator: FitnessEvaluator,
        bounds_lo: NDArray[np.float64],
        bounds_hi: NDArray[np.float64],
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iters: int = 500,
        tol: float = 1e-8,
        patience: int = 50,
        bounds_policy: Optional[BoundsPolicy] = None,
        topology: Optional[Topology] = None,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.evaluator = evaluator
        self.bounds_lo = np.asarray(bounds_lo, dtype=float)
        self.bounds_hi = np.asarray(bounds_hi, dtype=float)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.tol = tol
        self.patience = patience
        self.bounds_policy = bounds_policy or ClampBounds()
        self.topology = topology or GlobalBestTopology()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.log = logger or logging.getLogger(__name__)
        self.vmax = (self.bounds_hi - self.bounds_lo) * 0.5

    def run(self, on_iteration: Optional[Callable] = None) -> RunResult:
        swarm = build_swarm(self.n_particles, self.bounds_lo, self.bounds_hi, self.rng)

        t_start = time.perf_counter()
        time_eval = 0.0
        time_update = 0.0
        history: List[float] = []
        stagnation = 0

        self.evaluator.open()

        t0 = time.perf_counter()
        fitnesses = self.evaluator.evaluate(swarm.get_positions())
        time_eval += time.perf_counter() - t0
        swarm.update_global_best(fitnesses)
        history.append(swarm.best_fitness)

        if on_iteration is not None:
            on_iteration(0, swarm.get_positions(), swarm.best_position.copy(), swarm.best_fitness)

        final_iter = 0
        for it in range(1, self.max_iters + 1):
            final_iter = it

            t0 = time.perf_counter()
            for i, particle in enumerate(swarm.particles):
                gbest = self.topology.get_best(i, swarm)
                particle.update_velocity(gbest, self.w, self.c1, self.c2, self.rng, self.vmax)
                particle.update_position()
                particle.position, particle.velocity = self.bounds_policy.apply(
                    particle.position, particle.velocity, self.bounds_lo, self.bounds_hi
                )
            time_update += time.perf_counter() - t0

            t0 = time.perf_counter()
            fitnesses = self.evaluator.evaluate(swarm.get_positions())
            time_eval += time.perf_counter() - t0

            prev_best = swarm.best_fitness
            swarm.update_global_best(fitnesses)
            history.append(swarm.best_fitness)

            improvement = prev_best - swarm.best_fitness
            stagnation = stagnation + 1 if improvement < self.tol else 0

            self.log.info(
                "iter=%d best=%.6e t_eval_ms=%.2f t_update_ms=%.2f",
                it,
                swarm.best_fitness,
                time_eval * 1000,
                time_update * 1000,
            )

            if on_iteration is not None:
                on_iteration(it, swarm.get_positions(), swarm.best_position.copy(), swarm.best_fitness)

            if stagnation >= self.patience:
                self.log.info("early_stop iter=%d reason=stagnation", it)
                break

        self.evaluator.close()
        t_total = time.perf_counter() - t_start

        return RunResult(
            best_position=swarm.best_position,
            best_fitness=swarm.best_fitness,
            iterations=final_iter,
            time_total=t_total,
            time_eval=time_eval,
            time_update=time_update,
            history=history,
        )
