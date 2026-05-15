import time

import numpy as np

from core.pso import PSO
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadPoolEvaluator
from parallel.multiprocessing_eval import ProcessPoolEvaluator
from parallel.asyncio_eval import AsyncioEvaluator
from parallel.numpy_eval import NumpyEvaluator
from objectives.sphere import sphere
from objectives.noisy_service import noisy_sphere
from objectives.vectorized import sphere_batch


def _make_pso(evaluator, seed=42, dim=10, n_particles=30, max_iters=200):
    lo = np.full(dim, -5.12)
    hi = np.full(dim, 5.12)
    return PSO(
        evaluator=evaluator,
        bounds_lo=lo,
        bounds_hi=hi,
        n_particles=n_particles,
        max_iters=max_iters,
        seed=seed,
    )


def test_reproducibility():
    r1 = _make_pso(SequentialEvaluator(sphere), seed=0).run()
    r2 = _make_pso(SequentialEvaluator(sphere), seed=0).run()
    assert r1.best_fitness == r2.best_fitness


def test_bounds_respected():
    lo = np.full(10, -5.12)
    hi = np.full(10, 5.12)
    positions_seen = []
    pso = _make_pso(SequentialEvaluator(sphere), seed=1)
    pso.run(on_iteration=lambda it, pos, bp, bf: positions_seen.extend(pos))
    for p in positions_seen:
        assert np.all(p >= lo - 1e-10)
        assert np.all(p <= hi + 1e-10)


def test_monotonicity():
    result = _make_pso(SequentialEvaluator(sphere)).run()
    for i in range(1, len(result.history)):
        assert result.history[i] <= result.history[i - 1] + 1e-12


def test_convergence_sphere():
    result = _make_pso(SequentialEvaluator(sphere), max_iters=500).run()
    assert result.best_fitness < 1e-4


def test_v1_matches_v0():
    r0 = _make_pso(SequentialEvaluator(sphere)).run()
    r1 = _make_pso(ThreadPoolEvaluator(sphere)).run()
    assert r0.best_fitness == r1.best_fitness


def test_v2_matches_v0():
    r0 = _make_pso(SequentialEvaluator(sphere)).run()
    r2 = _make_pso(ProcessPoolEvaluator(sphere)).run()
    assert r0.best_fitness == r2.best_fitness


def test_v3_matches_v0():
    r0 = _make_pso(SequentialEvaluator(sphere)).run()
    r3 = _make_pso(AsyncioEvaluator(sphere)).run()
    assert r0.best_fitness == r3.best_fitness


def test_v4_matches_v0():
    r0 = _make_pso(SequentialEvaluator(sphere)).run()
    r4 = _make_pso(NumpyEvaluator(sphere_batch)).run()
    assert r0.best_fitness == r4.best_fitness


def test_v3_speedup_on_noisy():
    kwargs = dict(dim=2, n_particles=10, max_iters=3)

    t0 = time.perf_counter()
    _make_pso(SequentialEvaluator(noisy_sphere), **kwargs).run()
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    _make_pso(AsyncioEvaluator(noisy_sphere), **kwargs).run()
    t_async = time.perf_counter() - t0

    assert t_async < t_seq
