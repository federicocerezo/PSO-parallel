import numpy as np
import pytest

from objectives.wind_farm import WindFarm

D = WindFarm.D  # 80 m


def test_single_turbine_no_wake():
    """A single turbine has no wake partners → zero wake loss and zero penalty."""
    wf = WindFarm(n_turbines=1)
    x = np.array([1000.0, 1000.0])
    assert wf(x) == pytest.approx(0.0, abs=1e-10)


def test_two_aligned_downstream_has_wake():
    """
    Turbine behind another (same y, higher x) must suffer a velocity
    deficit, so total wake loss > 0.
    """
    wf = WindFarm(n_turbines=2, penalty_coeff=0.0)
    # Wind from west (+x): turbine at x=200 is upstream of x=1200
    x = np.array([200.0, 1000.0, 1200.0, 1000.0])
    assert wf(x) > 0.0


def test_two_turbines_same_x_no_wake():
    """
    Turbines at the same downstream coordinate are not in each other's
    wake (no upstream/downstream relationship) → zero wake loss.
    Lateral separation is large enough to avoid the penalty too.
    """
    wf = WindFarm(n_turbines=2, penalty_coeff=0.0)
    # Both at x=500, separated 1400 m laterally (>> 3D=240 m)
    x = np.array([500.0, 300.0, 500.0, 1700.0])
    assert wf(x) == pytest.approx(0.0, abs=1e-10)


def test_proximity_penalty_activated():
    """
    Two turbines separated by 50 m (< 3D = 240 m) must produce a
    higher objective than the same layout without the penalty term.
    """
    x_close = np.array([1000.0, 1000.0, 1050.0, 1000.0])  # 50 m apart
    wf_no_pen = WindFarm(n_turbines=2, penalty_coeff=0.0)
    wf_with_pen = WindFarm(n_turbines=2, penalty_coeff=1e-4)
    assert wf_with_pen(x_close) > wf_no_pen(x_close)


def test_penalty_zero_when_well_separated():
    """Turbines separated by more than 3D must not trigger the penalty."""
    wf_no_pen = WindFarm(n_turbines=2, penalty_coeff=0.0)
    wf_with_pen = WindFarm(n_turbines=2, penalty_coeff=1e-4)
    # 500 m separation >> 3D = 240 m
    x = np.array([500.0, 1000.0, 1000.0, 1000.0])
    assert wf_with_pen(x) == pytest.approx(wf_no_pen(x), rel=1e-12)


def test_reproducibility():
    """Same input always yields the same output (deterministic for unidirectional wind)."""
    wf = WindFarm(n_turbines=5, seed=42)
    x = np.linspace(100.0, 1900.0, 10)
    assert wf(x) == wf(x)


def test_vectorized_matches_loop():
    """
    Vectorised __call__ and sequential _evaluate_loop must agree to
    floating-point precision.  This validates that the NumPy broadcasting
    in the vectorised path produces identical results to the explicit loops.
    """
    wf = WindFarm(n_turbines=5)
    x = np.linspace(100.0, 1900.0, 10)
    assert wf(x) == pytest.approx(wf._evaluate_loop(x), rel=1e-6)


def test_bounds_length():
    """bounds property must have 2*n_turbines entries."""
    for n in (1, 5, 15):
        wf = WindFarm(n_turbines=n)
        assert len(wf.bounds) == 2 * n


def test_bounds_range():
    """Every bound entry must be (0, terrain_size)."""
    wf = WindFarm(n_turbines=5, terrain_size=2000.0)
    for lo, hi in wf.bounds:
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(2000.0)
