from __future__ import annotations
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt


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
