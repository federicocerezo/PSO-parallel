from __future__ import annotations
from typing import List
import numpy as np
from numpy.typing import NDArray

from .particle import Particle


def build_swarm(
    n_particles: int,
    bounds_lo: NDArray[np.float64],
    bounds_hi: NDArray[np.float64],
    rng: np.random.Generator,
) -> "Swarm":
    span = bounds_hi - bounds_lo
    particles = []
    for _ in range(n_particles):
        position = rng.uniform(bounds_lo, bounds_hi)
        velocity = rng.uniform(-span * 0.2, span * 0.2)
        particles.append(Particle(position, velocity))
    return Swarm(particles)


class Swarm:
    def __init__(self, particles: List[Particle]):
        self.particles = particles
        self.best_position: NDArray[np.float64] = particles[0].position.copy()
        self.best_fitness: float = float("inf")

    def get_positions(self) -> List[NDArray[np.float64]]:
        return [p.position.copy() for p in self.particles]

    def update_global_best(self, fitnesses: List[float]) -> None:
        for particle, fitness in zip(self.particles, fitnesses):
            particle.update_best(fitness)
            if particle.best_fitness < self.best_fitness:
                self.best_fitness = particle.best_fitness
                self.best_position = particle.best_position.copy()
