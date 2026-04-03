import numpy as np
from numpy.typing import NDArray


def ackley(x: NDArray[np.float64], a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e)
