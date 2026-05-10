import time
import numpy as np
from numpy.typing import NDArray


def noisy_sphere(
    x: NDArray[np.float64],
    min_delay: float = 0.005,
    max_delay: float = 0.05,
) -> float:
    """
    Sphere function with simulated asymmetric I/O latency.

    Each call sleeps a random duration in [min_delay, max_delay] seconds,
    modelling a fitness evaluation that queries an external service (e.g.
    a REST endpoint, a database, or a simulator).  The latency is
    intentionally asymmetric per particle so that asyncio.gather produces
    a measurable speedup over sequential evaluation: sequential time ≈
    sum(latencies), while concurrent time ≈ max(latencies).

    Thread-safe: numpy Generator is created per-call so concurrent
    invocations from run_in_executor do not share mutable state.
    """
    rng = np.random.default_rng()
    delay = float(rng.uniform(min_delay, max_delay))
    time.sleep(delay)
    return float(np.sum(x ** 2))
