from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.typing import NDArray


def plot_convergence(
    history: List[float],
    title: str = "Convergence",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def animate_swarm_2d(
    frames: List[Tuple[NDArray, NDArray, float]],
    objective: Callable,
    bounds: Tuple[float, float],
    history: List[float],
    title: str = "PSO",
    save_path: Optional[str] = None,
    fps: int = 10,
) -> None:
    lo, hi = bounds
    grid_n = 200
    xs = np.linspace(lo, hi, grid_n)
    ys = np.linspace(lo, hi, grid_n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[objective(np.array([X[i, j], Y[i, j]])) for j in range(grid_n)] for i in range(grid_n)])

    fig, (ax_swarm, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    ax_swarm.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax_swarm.contour(X, Y, Z, levels=30, colors="white", linewidths=0.3, alpha=0.4)
    ax_swarm.set_xlim(lo, hi)
    ax_swarm.set_ylim(lo, hi)
    ax_swarm.set_xlabel("x₀")
    ax_swarm.set_ylabel("x₁")

    scat = ax_swarm.scatter([], [], c="red", s=20, zorder=5, label="particles")
    gbest_dot, = ax_swarm.plot([], [], "w*", markersize=14, zorder=6, label="global best")
    iter_text = ax_swarm.text(0.02, 0.97, "", transform=ax_swarm.transAxes,
                               color="white", fontsize=9, va="top")
    ax_swarm.legend(loc="lower right", fontsize=8)

    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best fitness (log scale)")
    ax_conv.set_yscale("log")
    ax_conv.grid(True, which="both", ls="--", alpha=0.4)
    ax_conv.set_xlim(0, len(history))
    pos_vals = [v for v in history if v > 0]
    if pos_vals:
        ax_conv.set_ylim(min(pos_vals) * 0.5, max(history) * 2)
    conv_line, = ax_conv.plot([], [], "b-", linewidth=1.5)
    conv_dot, = ax_conv.plot([], [], "ro", markersize=5)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        gbest_dot.set_data([], [])
        conv_line.set_data([], [])
        conv_dot.set_data([], [])
        iter_text.set_text("")
        return scat, gbest_dot, conv_line, conv_dot, iter_text

    def update(frame_idx):
        positions, gbest, fitness = frames[frame_idx]
        scat.set_offsets(positions[:, :2])
        gbest_dot.set_data([gbest[0]], [gbest[1]])
        iters = list(range(frame_idx + 1))
        conv_line.set_data(iters, history[:frame_idx + 1])
        conv_dot.set_data([frame_idx], [history[frame_idx]])
        iter_text.set_text(f"iter={frame_idx}  best={fitness:.3e}")
        return scat, gbest_dot, conv_line, conv_dot, iter_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=1000 // fps, blit=True
    )

    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
    else:
        plt.show()
