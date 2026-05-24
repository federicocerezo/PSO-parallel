import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.pso import PSO
from parallel.sequential import SequentialEvaluator
from objectives import REGISTRY
from viz.plots import plot_convergence, animate_swarm_2d, animate_swarm_3d


def main():
    parser = argparse.ArgumentParser(description="Generate PSO visualizations")
    parser.add_argument("--objective", default="sphere", choices=list(REGISTRY.keys()))
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n-particles", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save-dir", default="results/viz")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    obj_info = REGISTRY[args.objective]
    objective = obj_info["fn"]
    lo = np.full(args.dim, obj_info["bounds"][0])
    hi = np.full(args.dim, obj_info["bounds"][1])

    frames = []

    def on_iteration(it, positions, gbest, fitness):
        frames.append((np.array(positions), gbest.copy(), fitness))

    evaluator = SequentialEvaluator(objective)
    pso = PSO(
        evaluator=evaluator,
        bounds_lo=lo,
        bounds_hi=hi,
        n_particles=args.n_particles,
        max_iters=args.max_iters,
        seed=args.seed,
    )

    result = pso.run(on_iteration=on_iteration)

    title = f"{args.objective} d={args.dim} n={args.n_particles} seed={args.seed}"

    conv_path = os.path.join(args.save_dir, f"{args.objective}_d{args.dim}_convergence.png")
    plot_convergence(result.history, title=title, save_path=conv_path)
    print(f"Convergence plot saved to {conv_path}")

    if args.dim == 2:
        gif_path = os.path.join(args.save_dir, f"{args.objective}_d{args.dim}_animation.gif")
        animate_swarm_2d(
            frames=frames,
            objective=objective,
            bounds=obj_info["bounds"],
            history=result.history,
            title=title,
            save_path=gif_path,
            fps=args.fps,
        )
        print(f"Animation saved to {gif_path}")
    elif args.dim == 3:
        gif_path = os.path.join(args.save_dir, f"{args.objective}_d{args.dim}_animation.gif")
        animate_swarm_3d(
            frames=frames,
            history=result.history,
            title=title,
            save_path=gif_path,
            fps=args.fps,
        )
        print(f"3D animation saved to {gif_path}")

    print(f"Best fitness: {result.best_fitness:.6e} in {result.iterations} iterations")


if __name__ == "__main__":
    main()
