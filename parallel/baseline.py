import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from pyswarm import pso


class BaselineResult:
    def __init__(
        self,
        best_position: NDArray[np.float64],
        best_fitness: float,
        time_total: float,
    ):
        self.best_position = best_position
        self.best_fitness = best_fitness
        self.time_total = time_total


def run_pyswarm(
    objective: Callable[[NDArray[np.float64]], float],
    bounds_lo: NDArray[np.float64],
    bounds_hi: NDArray[np.float64],
    n_particles: int = 30,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    max_iters: int = 500,
) -> BaselineResult:
    lo = bounds_lo.tolist()
    hi = bounds_hi.tolist()

    t0 = time.perf_counter()
    best_pos, best_fit = pso(
        objective,
        lb=lo,
        ub=hi,
        swarmsize=n_particles,
        omega=w,
        phip=c1,
        phig=c2,
        maxiter=max_iters,
        debug=False,
    )
    t_total = time.perf_counter() - t0

    return BaselineResult(
        best_position=np.asarray(best_pos),
        best_fitness=float(best_fit),
        time_total=t_total,
    )
