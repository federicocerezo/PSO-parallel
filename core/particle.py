from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class Particle:
    def __init__(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
    ):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()
        self.best_fitness: float = float("inf")

    def update_velocity(
        self,
        gbest: NDArray[np.float64],
        w: float,
        c1: float,
        c2: float,
        rng: np.random.Generator,
        vmax: NDArray[np.float64],
    ) -> None:
        r1 = rng.random(self.position.shape)
        r2 = rng.random(self.position.shape)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -vmax, vmax)

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    def update_best(self, fitness: float) -> None:
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
