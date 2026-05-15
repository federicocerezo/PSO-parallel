import numpy as np
from numpy.typing import NDArray


def sphere_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(X ** 2, axis=1)


def rosenbrock_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def rastrigin_batch(X: NDArray[np.float64]) -> NDArray[np.float64]:
    n = X.shape[1]
    return 10.0 * n + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def ackley_batch(
    X: NDArray[np.float64],
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2 * np.pi,
) -> NDArray[np.float64]:
    n = X.shape[1]
    sum_sq = np.sum(X ** 2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e
