import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objectives import REGISTRY
from experiments.runner import RunConfig, run_experiment

EVALUATORS = ["sequential", "threading", "multiprocessing"]


def main():
    parser = argparse.ArgumentParser(description="Run PSO on a benchmark function")
    parser.add_argument("--objective", default="sphere", choices=list(REGISTRY.keys()))
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--w", type=float, default=0.7)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluator", default="sequential", choices=EVALUATORS)
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()

    obj_info = REGISTRY[args.objective]

    config = RunConfig(
        objective=args.objective,
        dim=args.dim,
        bounds_lo=obj_info["bounds"][0],
        bounds_hi=obj_info["bounds"][1],
        n_particles=args.n_particles,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        max_iters=args.max_iters,
        patience=args.patience,
        seed=args.seed,
        evaluator=args.evaluator,
        save_dir="" if args.no_save else args.save_dir,
        compare_baseline=args.compare_baseline,
    )

    result = run_experiment(config)

    labels = {
        "sequential": "V0 (sequential)",
        "threading": "V1 (threading)",
        "multiprocessing": "V2 (multiprocessing)",
    }
    label = labels[args.evaluator]
    pct_eval = result.time_eval / result.time_total * 100 if result.time_total > 0 else 0
    pct_update = result.time_update / result.time_total * 100 if result.time_total > 0 else 0

    print(f"{'─'*44}")
    print(f"  {label}  |  {args.objective} d={args.dim} seed={args.seed}")
    print(f"{'─'*44}")
    print(f"  Best fitness : {result.best_fitness:.6e}")
    print(f"  Iterations   : {result.iterations}")
    print(f"  Time total   : {result.time_total:.3f}s")
    print(f"  Time eval    : {result.time_eval:.3f}s ({pct_eval:.1f}%)")
    print(f"  Time update  : {result.time_update:.3f}s ({pct_update:.1f}%)")

    baseline = getattr(result, "_baseline", None)
    if baseline is not None:
        print(f"{'─'*44}")
        print(f"  pyswarm baseline")
        print(f"{'─'*44}")
        print(f"  Best fitness : {baseline.best_fitness:.6e}")
        print(f"  Time total   : {baseline.time_total:.3f}s")
        print(f"{'─'*44}")
        winner = label if result.best_fitness <= baseline.best_fitness else "pyswarm"
        print(f"  Winner (fitness): {winner}")
    print(f"{'─'*44}")


if __name__ == "__main__":
    main()
