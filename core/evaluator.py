from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np
from numpy.typing import NDArray


class FitnessEvaluator(ABC):
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        ...


class SequentialEvaluator(FitnessEvaluator):
    def __init__(self, objective: Callable[[NDArray[np.float64]], float]):
        self.objective = objective

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        return [float(self.objective(x)) for x in positions]
