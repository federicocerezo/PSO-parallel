from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BoundsPolicy(ABC):
    @abstractmethod
    def apply(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        lo: NDArray[np.float64],
        hi: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ...


class ClampBounds(BoundsPolicy):
    def apply(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        lo: NDArray[np.float64],
        hi: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        clipped = np.clip(position, lo, hi)
        hit = (clipped == lo) | (clipped == hi)
        new_vel = velocity.copy()
        new_vel[hit] = 0.0
        return clipped, new_vel
