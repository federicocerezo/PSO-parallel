from typing import Callable, List

import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


class NumpyEvaluator(FitnessEvaluator):
    def __init__(self, batch_objective: Callable[[NDArray[np.float64]], NDArray[np.float64]]):
        self.batch_objective = batch_objective

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        X = np.stack(positions)
        return self.batch_objective(X).tolist()
