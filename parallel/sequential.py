from typing import Callable, List
import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


class SequentialEvaluator(FitnessEvaluator):
    def __init__(self, objective: Callable[[NDArray[np.float64]], float]):
        self.objective = objective

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        return [float(self.objective(x)) for x in positions]
