import time
import numpy as np
from numpy.typing import NDArray


def noisy_sphere(
    x: NDArray[np.float64],
    min_delay: float = 0.005,
    max_delay: float = 0.05,
) -> float:
    rng = np.random.default_rng()
    delay = float(rng.uniform(min_delay, max_delay))
    time.sleep(delay)
    return float(np.sum(x ** 2))
