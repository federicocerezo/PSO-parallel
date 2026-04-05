from abc import ABC, abstractmethod
from typing import List
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
