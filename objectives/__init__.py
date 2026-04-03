from .sphere import sphere
from .ackley import ackley
from .rosenbrock import rosenbrock
from .rastrigin import rastrigin

REGISTRY = {
    "sphere": {
        "fn": sphere,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
    "ackley": {
        "fn": ackley,
        "bounds": (-32.768, 32.768),
        "optimum": 0.0,
    },
    "rosenbrock": {
        "fn": rosenbrock,
        "bounds": (-2.048, 2.048),
        "optimum": 0.0,
    },
    "rastrigin": {
        "fn": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
}
