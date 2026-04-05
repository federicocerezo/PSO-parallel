from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


def _evaluate_batch(
    objective: Callable[[NDArray[np.float64]], float],
    batch: List[NDArray[np.float64]],
) -> List[float]:
    return [float(objective(x)) for x in batch]


def _chunked(items: List, batch_size: int) -> List[List]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class ProcessPoolEvaluator(FitnessEvaluator):
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.objective = objective
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._executor: Optional[ProcessPoolExecutor] = None

    def open(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        if self._executor is None:
            self.open()

        n_workers = self.max_workers or 4
        batch_size = self.batch_size or max(1, len(positions) // (n_workers * 2))
        batches = _chunked(positions, batch_size)

        futures = [
            self._executor.submit(_evaluate_batch, self.objective, batch)
            for batch in batches
        ]

        results = []
        for future in futures:
            results.extend(future.result())
        return results
