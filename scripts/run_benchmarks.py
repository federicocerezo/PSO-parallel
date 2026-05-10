import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectives import REGISTRY
from experiments.runner import RunConfig, run_experiment

DIMS = [2, 10, 30]
SEEDS = [42, 43, 44]
EVALUATORS = ["sequential", "threading", "multiprocessing", "asyncio"]


def main():
    parser = argparse.ArgumentParser(description="Run PSO benchmark suite")
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--evaluator", default="sequential", choices=EVALUATORS)
    args = parser.parse_args()

    header = f"{'objective':12s} {'dim':>4} {'seed':>5} {'evaluator':>12} {'best_fitness':>14} {'time':>8}"
    print(header)
    print("-" * len(header))

    for objective in REGISTRY:
        for dim in DIMS:
            for seed in SEEDS:
                obj_info = REGISTRY[objective]
                config = RunConfig(
                    objective=objective,
                    dim=dim,
                    bounds_lo=obj_info["bounds"][0],
                    bounds_hi=obj_info["bounds"][1],
                    n_particles=args.n_particles,
                    max_iters=args.max_iters,
                    seed=seed,
                    evaluator=args.evaluator,
                    save_dir=args.save_dir,
                )
                result = run_experiment(config)
                print(
                    f"{objective:12s} {dim:>4d} {seed:>5d} {args.evaluator:>12s} "
                    f"{result.best_fitness:>14.4e} {result.time_total:>7.2f}s"
                )

    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
