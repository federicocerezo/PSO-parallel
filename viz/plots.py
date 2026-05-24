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


def animate_swarm_3d(
    frames: List[Tuple[NDArray, NDArray, float]],
    history: List[float],
    title: str = "PSO",
    save_path: Optional[str] = None,
    fps: int = 10,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    all_pos = np.vstack([f[0] for f in frames])
    lo = all_pos.min(axis=0)
    hi = all_pos.max(axis=0)
    pad = (hi - lo) * 0.05 + 1e-6

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title)

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_xlabel("x₀")
    ax3d.set_ylabel("x₁")
    ax3d.set_zlabel("x₂")
    ax3d.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax3d.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax3d.set_zlim(lo[2] - pad[2], hi[2] + pad[2])

    scat = ax3d.scatter([], [], [], c="red", s=20, depthshade=True, label="particles")
    gbest_dot = ax3d.scatter([], [], [], c="gold", s=200, marker="*", depthshade=False, label="global best")
    iter_text = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes, fontsize=9, va="top")
    ax3d.legend(loc="lower left", fontsize=8)

    ax_conv = fig.add_subplot(122)
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best fitness (log scale)")
    ax_conv.set_yscale("log")
    ax_conv.set_xlim(0, len(history))
    pos_vals = [v for v in history if v > 0]
    if pos_vals:
        ax_conv.set_ylim(min(pos_vals) * 0.5, max(history) * 2)
    ax_conv.grid(True, which="both", ls="--", alpha=0.4)
    conv_line, = ax_conv.plot([], [], "b-", linewidth=1.5)
    conv_dot, = ax_conv.plot([], [], "ro", markersize=5)

    def init():
        scat._offsets3d = (np.array([]), np.array([]), np.array([]))
        gbest_dot._offsets3d = (np.array([]), np.array([]), np.array([]))
        conv_line.set_data([], [])
        conv_dot.set_data([], [])
        iter_text.set_text("")
        return scat, gbest_dot, conv_line, conv_dot, iter_text

    def update(frame_idx):
        positions, gbest, fitness = frames[frame_idx]
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        gbest_dot._offsets3d = (np.array([gbest[0]]), np.array([gbest[1]]), np.array([gbest[2]]))
        iters = list(range(frame_idx + 1))
        conv_line.set_data(iters, history[:frame_idx + 1])
        conv_dot.set_data([frame_idx], [history[frame_idx]])
        iter_text.set_text(f"iter={frame_idx}  best={fitness:.3e}")
        return scat, gbest_dot, conv_line, conv_dot, iter_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init,
        interval=1000 // fps, blit=False
    )

    plt.tight_layout()
    if save_path:
        ani.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
    else:
        plt.show()
