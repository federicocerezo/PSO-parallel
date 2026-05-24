"""
Case study: wind farm layout optimisation with PSO.

Optimises the (x, y) positions of N turbines on a 2000 x 2000 m terrain
to minimise annual-energy-production wake losses (Jensen model, Katić
superposition).  Dimension d = 2N, so:

    --n-turbines 1   →  d=2
    --n-turbines 5   →  d=10   (default)
    --n-turbines 15  →  d=30

After the run, saves a PNG showing the optimal turbine layout with
Jensen wake cones.

Usage
-----
    python3 scripts/run_wind_farm.py
    python3 scripts/run_wind_farm.py --n-turbines 15 --max-iters 500
    python3 scripts/run_wind_farm.py --evaluator multiprocessing --n-particles 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from core.pso import PSO
from objectives.wind_farm import WindFarm
from parallel.asyncio_eval import AsyncioEvaluator
from parallel.multiprocessing_eval import ProcessPoolEvaluator
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadPoolEvaluator

EVALUATORS = ["sequential", "threading", "multiprocessing", "asyncio"]


def plot_layout(
    wf: WindFarm,
    best_pos: np.ndarray,
    best_fitness: float,
    save_path: str,
) -> None:
    """Save a layout plot: terrain, Jensen wake cones and turbine discs."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.add_patch(patches.Rectangle(
        (0, 0), wf.terrain_size, wf.terrain_size,
        lw=2, edgecolor="black", facecolor="#f5f5f5", zorder=0,
    ))

    pos = best_pos.reshape(-1, 2)

    # Jensen wake cones (wind from west = +x direction)
    for x0, y0 in pos:
        reach = wf.terrain_size - x0
        if reach > 0:
            half_w = wf.D / 2 + wf.k * reach
            cone = plt.Polygon(
                [[x0, y0], [x0 + reach, y0 + half_w], [x0 + reach, y0 - half_w]],
                alpha=0.07, color="orange", zorder=1,
            )
            ax.add_patch(cone)

    # Turbine rotor discs and labels
    for i, (x, y) in enumerate(pos):
        ax.add_patch(plt.Circle((x, y), wf.D / 2, color="#1976D2", alpha=0.85, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=4)

    # Wind direction arrow
    arrow_y = wf.terrain_size * 0.04
    ax.annotate(
        "", xy=(380, arrow_y), xytext=(60, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="#666", lw=2),
    )
    ax.text(220, arrow_y + 55, "wind →", fontsize=9, color="#666", ha="center")

    ax.set_xlim(-100, wf.terrain_size + 100)
    ax.set_ylim(-100, wf.terrain_size + 100)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Wind farm layout — {wf.n_turbines} turbines "
        f"(D={wf.D:.0f} m, Ct={wf.Ct}, k={wf.k})\n"
        f"Wake loss = {best_fitness:.4f}  |  "
        f"Terrain {wf.terrain_size:.0f} × {wf.terrain_size:.0f} m",
        fontsize=11,
    )
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Layout plot saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSO wind farm layout optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-turbines", type=int, default=5, choices=[1, 5, 15],
        help="Number of turbines  (1→d=2 | 5→d=10 | 15→d=30)",
    )
    parser.add_argument("--evaluator", default="sequential", choices=EVALUATORS)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="results/wind_farm")
    args = parser.parse_args()

    wf = WindFarm(n_turbines=args.n_turbines)
    dim = 2 * args.n_turbines
    lo = np.zeros(dim)
    hi = np.full(dim, wf.terrain_size)

    evaluator_map = {
        "sequential": SequentialEvaluator(wf),
        "threading": ThreadPoolEvaluator(wf),
        "multiprocessing": ProcessPoolEvaluator(wf),
        "asyncio": AsyncioEvaluator(wf),
    }
    evaluator = evaluator_map[args.evaluator]

    pso = PSO(
        evaluator=evaluator,
        bounds_lo=lo,
        bounds_hi=hi,
        n_particles=args.n_particles,
        max_iters=args.max_iters,
        patience=args.patience,
        seed=args.seed,
    )

    print("=" * 50)
    print("  Wind Farm Layout Optimisation — PSO")
    print("=" * 50)
    print(f"  Turbines     : {wf.n_turbines}  (d = {dim})")
    print(f"  Terrain      : {wf.terrain_size:.0f} × {wf.terrain_size:.0f} m")
    print(f"  D={wf.D:.0f} m  Ct={wf.Ct}  k={wf.k}  V∞={wf.v_inf} m/s")
    print(f"  Evaluator    : {args.evaluator}")
    print(f"  Particles    : {args.n_particles}   Max iters : {args.max_iters}")
    print(f"  Seed         : {args.seed}")
    print("=" * 50)

    result = pso.run()

    pct_eval = result.time_eval / result.time_total * 100 if result.time_total > 0 else 0
    print(f"  Best wake loss : {result.best_fitness:.6f}")
    print(f"  Iterations     : {result.iterations}")
    print(f"  Time total     : {result.time_total:.3f} s")
    print(f"  Time eval      : {result.time_eval:.3f} s  ({pct_eval:.1f} %)")
    print("=" * 50)
    print("  Optimal turbine positions:")
    for i, (x, y) in enumerate(result.best_position.reshape(-1, 2)):
        print(f"    T{i:>2d}  x={x:7.1f} m   y={y:7.1f} m")
    print("=" * 50)

    os.makedirs(args.save_dir, exist_ok=True)
    plot_layout(
        wf,
        result.best_position,
        result.best_fitness,
        os.path.join(args.save_dir, f"layout_N{args.n_turbines}_s{args.seed}.png"),
    )


if __name__ == "__main__":
    main()
