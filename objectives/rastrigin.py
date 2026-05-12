import numpy as np
from numpy.typing import NDArray


def rastrigin(x: NDArray[np.float64]) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))
