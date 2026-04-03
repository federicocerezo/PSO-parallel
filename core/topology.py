from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Topology(ABC):
    @abstractmethod
    def get_best(self, particle_idx: int, swarm: object) -> NDArray[np.float64]:
        ...


class GlobalBestTopology(Topology):
    def get_best(self, particle_idx: int, swarm: object) -> NDArray[np.float64]:
        return swarm.best_position.copy()
