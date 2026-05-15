from .sphere import sphere
from .ackley import ackley
from .rosenbrock import rosenbrock
from .rastrigin import rastrigin
from .noisy_service import noisy_sphere
from .vectorized import sphere_batch, ackley_batch, rosenbrock_batch, rastrigin_batch

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
    "noisy_sphere": {
        "fn": noisy_sphere,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
    },
}

BATCH_REGISTRY = {
    "sphere": sphere_batch,
    "ackley": ackley_batch,
    "rosenbrock": rosenbrock_batch,
    "rastrigin": rastrigin_batch,
}
