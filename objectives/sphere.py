import numpy as np
from numpy.typing import NDArray


def sphere(x: NDArray[np.float64]) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))
