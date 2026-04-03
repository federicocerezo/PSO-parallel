import numpy as np

from core.pso import PSO
from core.evaluator import SequentialEvaluator
from objectives.sphere import sphere


def make_pso(seed=42, dim=2, n_particles=20, max_iters=200, objective=sphere, **kwargs):
    lo = np.full(dim, -5.12)
    hi = np.full(dim, 5.12)
    evaluator = SequentialEvaluator(objective)
    return PSO(
        evaluator=evaluator,
        bounds_lo=lo,
        bounds_hi=hi,
        n_particles=n_particles,
        max_iters=max_iters,
        seed=seed,
        **kwargs,
    )


def test_reproducibility():
    r1 = make_pso(seed=0).run()
    r2 = make_pso(seed=0).run()
    assert r1.best_fitness == r2.best_fitness
    assert np.allclose(r1.history, r2.history)


def test_different_seeds_differ():
    r1 = make_pso(seed=0).run()
    r2 = make_pso(seed=99).run()
    assert r1.best_fitness != r2.best_fitness


def test_bounds_respected():
    lo = np.array([-5.12, -5.12])
    hi = np.array([5.12, 5.12])
    evaluator = SequentialEvaluator(sphere)
    pso = PSO(evaluator=evaluator, bounds_lo=lo, bounds_hi=hi, n_particles=20, max_iters=50, seed=1)

    positions_seen = []

    def on_iter(it, positions, best_position, best_fitness):
        positions_seen.extend(positions)

    pso.run(on_iteration=on_iter)

    for pos in positions_seen:
        assert np.all(pos >= lo - 1e-10)
        assert np.all(pos <= hi + 1e-10)


def test_monotonicity():
    result = make_pso(seed=42, max_iters=100).run()
    for i in range(1, len(result.history)):
        assert result.history[i] <= result.history[i - 1] + 1e-12


def test_history_length():
    result = make_pso(seed=42, max_iters=100, patience=200).run()
    assert len(result.history) == result.iterations + 1


def test_convergence_sphere_2d():
    result = make_pso(seed=42, dim=2, n_particles=30, max_iters=500).run()
    assert result.best_fitness < 1e-4


def test_convergence_sphere_10d():
    lo = np.full(10, -5.12)
    hi = np.full(10, 5.12)
    evaluator = SequentialEvaluator(sphere)
    pso = PSO(evaluator=evaluator, bounds_lo=lo, bounds_hi=hi, n_particles=50, max_iters=1000, seed=42)
    result = pso.run()
    assert result.best_fitness < 1.0


def test_initial_snapshot_included():
    snapshots = []

    def on_iter(it, positions, best_position, best_fitness):
        snapshots.append(it)

    result = make_pso(seed=0, max_iters=10, patience=100).run(on_iteration=on_iter)
    assert 0 in snapshots
    assert len(snapshots) == result.iterations + 1
