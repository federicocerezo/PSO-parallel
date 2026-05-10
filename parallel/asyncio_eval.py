import asyncio
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


class AsyncioEvaluator(FitnessEvaluator):
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        max_concurrent: Optional[int] = None,
    ):
        self.objective = objective
        self.max_concurrent = max_concurrent

    async def _eval_one(
        self,
        x: NDArray[np.float64],
        semaphore: Optional[asyncio.Semaphore],
    ) -> float:
        loop = asyncio.get_running_loop()
        if semaphore is not None:
            async with semaphore:
                return await loop.run_in_executor(None, self.objective, x)
        return await loop.run_in_executor(None, self.objective, x)

    async def _eval_all(self, positions: List[NDArray[np.float64]]) -> List[float]:
        semaphore = asyncio.Semaphore(self.max_concurrent) if self.max_concurrent else None
        tasks = [self._eval_one(x, semaphore) for x in positions]
        return list(await asyncio.gather(*tasks))

    def evaluate(self, positions: List[NDArray[np.float64]]) -> List[float]:
        return asyncio.run(self._eval_all(positions))
