import numpy as np
from numpy.typing import NDArray


def rosenbrock(x: NDArray[np.float64]) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
