from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


class ThreadPoolEvaluator(FitnessEvaluator):
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        max_workers: Optional[int] = None,
    ):
        self.objective = objective
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    def open(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        if self._executor is None:
            self.open()
        return list(self._executor.map(self.objective, positions))
