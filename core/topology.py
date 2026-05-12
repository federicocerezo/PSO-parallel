from abc import ABC, abstractmethod


class Topology(ABC):
    @abstractmethod
    def get_best(self, particle_idx: int, swarm: object) -> object:
        ...


class GlobalBestTopology(Topology):
    def get_best(self, particle_idx: int, swarm: object) -> object:
        return swarm.best_position.copy()
