import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objectives import REGISTRY
from experiments.grid_search import grid_search

W_VALUES = [0.4, 0.7, 0.9]
C1_VALUES = [1.0, 1.5, 2.0]
C2_VALUES = [1.0, 1.5, 2.0]
N_PARTICLES_VALUES = [20, 30]
SEEDS = [42, 43, 44, 45, 46]


def main():
    parser = argparse.ArgumentParser(description="PSO hyperparameter grid search")
    parser.add_argument("--objective", default="sphere", choices=list(REGISTRY.keys()))
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--save-dir", default="results/grid_search")
    args = parser.parse_args()

    n_combos = len(W_VALUES) * len(C1_VALUES) * len(C2_VALUES) * len(N_PARTICLES_VALUES)
    print(f"Grid search: {args.objective} d={args.dim}")
    print(f"Combinations: {n_combos}")
    print(f"Seeds per combo: {len(SEEDS)}\n")

    results = grid_search(
        objective=args.objective,
        dim=args.dim,
        w_values=W_VALUES,
        c1_values=C1_VALUES,
        c2_values=C2_VALUES,
        n_particles_values=N_PARTICLES_VALUES,
        max_iters=args.max_iters,
        seeds=SEEDS,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{args.objective}_d{args.dim}_grid.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTop 5 configurations for {args.objective} d={args.dim}:")
    print(f"{'w':>6} {'c1':>6} {'c2':>6} {'n':>6} {'mean_fitness':>14} {'std':>12}")
    for r in results[:5]:
        print(
            f"{r['w']:>6.2f} {r['c1']:>6.2f} {r['c2']:>6.2f} {r['n_particles']:>6d} "
            f"{r['mean_fitness']:>14.4e} {r['std_fitness']:>12.4e}"
        )

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
