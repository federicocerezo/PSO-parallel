import asyncio
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from core.evaluator import FitnessEvaluator


class AsyncioEvaluator(FitnessEvaluator):
    """
    V3: Cooperative concurrency via asyncio.gather + run_in_executor.

    When to use
    -----------
    This evaluator is effective when the objective function is I/O-bound
    (e.g. it calls a local service, reads a file, or sleeps waiting for a
    response).  asyncio.gather launches all particle evaluations
    concurrently: while one coroutine is blocked waiting for I/O, the
    event loop switches to another.  Total wall time ≈ max(latencies)
    instead of sum(latencies).

    When NOT to use
    ---------------
    For pure CPU-bound objectives (e.g. Sphere, Rastrigin computed in
    NumPy) asyncio gives no speedup: the work runs in threads that still
    compete for the GIL.  Use ProcessPoolEvaluator (V2) or the vectorised
    evaluator (V4) for those cases.

    Parameters
    ----------
    objective : callable
        Sync fitness function x -> float.  Must be thread-safe because
        run_in_executor dispatches it to the default ThreadPoolExecutor.
    max_concurrent : int, optional
        Cap on simultaneous coroutines via asyncio.Semaphore.  Useful to
        avoid overwhelming the target service.  None = no limit.
    """

    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        max_concurrent: Optional[int] = None,
    ):
        self.objective = objective
        self.max_concurrent = max_concurrent

    # open() / close() are no-ops: asyncio.run() manages its own event
    # loop internally, so there is nothing to set up or tear down here.

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
        """
        Synchronous entry point required by FitnessEvaluator.

        Internally spins up a fresh event loop via asyncio.run(), gathers
        all coroutines, and returns results in the same order as the input.
        A new event loop per PSO iteration is cheap relative to any
        realistic I/O latency.
        """
        return asyncio.run(self._eval_all(positions))
