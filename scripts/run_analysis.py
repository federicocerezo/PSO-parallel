import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    records = []
    for entry in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, entry, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)
        rec = {**data["config"], **data["result"]}
        records.append(rec)
    return records


def print_summary_table(records: List[Dict[str, Any]]) -> None:
    evaluators = sorted({r["evaluator"] for r in records})
    objectives = sorted({r["objective"] for r in records})
    dims = sorted({r["dim"] for r in records})

    header = f"{'objective':<14} {'dim':>4}  " + "  ".join(f"{e:>16}" for e in evaluators)
    print(header)
    print("-" * len(header))

    for obj in objectives:
        for dim in dims:
            row_parts = [f"{obj:<14} {dim:>4}"]
            for ev in evaluators:
                subset = [r for r in records if r["objective"] == obj and r["dim"] == dim and r["evaluator"] == ev]
                if not subset:
                    row_parts.append(f"{'—':>16}")
                    continue
                times = [r["time_total"] for r in subset]
                row_parts.append(f"{np.mean(times):>7.3f}s ± {np.std(times):.3f}")
            print("  ".join(row_parts))


def print_speedup_table(records: List[Dict[str, Any]]) -> None:
    evaluators = sorted({r["evaluator"] for r in records})
    objectives = sorted({r["objective"] for r in records})
    dims = sorted({r["dim"] for r in records})

    print("\nSpeedup vs sequential (mean time_total):")
    header = f"{'objective':<14} {'dim':>4}  " + "  ".join(f"{e:>12}" for e in evaluators)
    print(header)
    print("-" * len(header))

    for obj in objectives:
        for dim in dims:
            baseline = [r["time_total"] for r in records if r["objective"] == obj and r["dim"] == dim and r["evaluator"] == "sequential"]
            if not baseline:
                continue
            base_mean = np.mean(baseline)
            row_parts = [f"{obj:<14} {dim:>4}"]
            for ev in evaluators:
                subset = [r["time_total"] for r in records if r["objective"] == obj and r["dim"] == dim and r["evaluator"] == ev]
                if not subset:
                    row_parts.append(f"{'—':>12}")
                    continue
                speedup = base_mean / np.mean(subset)
                row_parts.append(f"{speedup:>11.2f}x")
            print("  ".join(row_parts))


def plot_fitness_boxplots(records: List[Dict[str, Any]], save_dir: str) -> None:
    evaluators = sorted({r["evaluator"] for r in records})
    objectives = sorted({r["objective"] for r in records})

    fig, axes = plt.subplots(1, len(objectives), figsize=(4 * len(objectives), 5), sharey=False)
    if len(objectives) == 1:
        axes = [axes]

    for ax, obj in zip(axes, objectives):
        data = []
        labels = []
        for ev in evaluators:
            fitnesses = [r["best_fitness"] for r in records if r["objective"] == obj and r["evaluator"] == ev]
            if fitnesses:
                data.append(fitnesses)
                labels.append(ev)
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_yscale("log")
        ax.set_title(obj)
        ax.set_ylabel("best fitness")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", ls="--", alpha=0.4)

    plt.suptitle("Final fitness distribution by evaluator")
    plt.tight_layout()
    path = os.path.join(save_dir, "boxplot_fitness.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Boxplot saved to {path}")


def plot_time_bars(records: List[Dict[str, Any]], save_dir: str) -> None:
    evaluators = sorted({r["evaluator"] for r in records})
    objectives = sorted({r["objective"] for r in records})
    dims = sorted({r["dim"] for r in records})

    combos = [(obj, dim) for obj in objectives for dim in dims]
    x = np.arange(len(combos))
    width = 0.8 / len(evaluators)

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.2), 5))

    for i, ev in enumerate(evaluators):
        means = []
        for obj, dim in combos:
            times = [r["time_total"] for r in records if r["objective"] == obj and r["dim"] == dim and r["evaluator"] == ev]
            means.append(np.mean(times) if times else 0.0)
        offset = (i - len(evaluators) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=ev)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{obj}\nd={dim}" for obj, dim in combos], fontsize=8)
    ax.set_ylabel("mean time (s)")
    ax.set_title("Time by evaluator, objective and dimension")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "bar_times.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Time bar chart saved to {path}")


def load_histories(results_dir: str) -> Dict:
    groups = defaultdict(list)
    for entry in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, entry, "summary.json")
        history_path = os.path.join(results_dir, entry, "history.csv")
        if not os.path.isfile(summary_path) or not os.path.isfile(history_path):
            continue
        with open(summary_path) as f:
            cfg = json.load(f)["config"]
        key = (cfg["objective"], cfg["dim"], cfg["evaluator"])
        history = []
        with open(history_path, newline="") as f:
            for row in csv.DictReader(f):
                history.append(float(row["best_fitness"]))
        groups[key].append(history)
    return dict(groups)


def plot_convergence_curves(groups: Dict, save_dir: str) -> None:
    objectives = sorted({k[0] for k in groups})
    dims = sorted({k[1] for k in groups})
    evaluators = sorted({k[2] for k in groups})
    colors = plt.cm.tab10.colors

    for obj in objectives:
        for dim in dims:
            if not any((obj, dim, ev) in groups for ev in evaluators):
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for i, ev in enumerate(evaluators):
                key = (obj, dim, ev)
                if key not in groups:
                    continue
                histories = groups[key]
                max_len = max(len(h) for h in histories)
                padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in histories])
                mean = padded.mean(axis=0)
                std = padded.std(axis=0)
                iters = np.arange(max_len)
                color = colors[i % len(colors)]
                ax.semilogy(iters, mean, label=ev, color=color)
                ax.fill_between(iters,
                                np.clip(mean - std, 1e-12, None),
                                mean + std,
                                alpha=0.2, color=color)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best fitness (log scale)")
            ax.set_title(f"{obj} d={dim} — convergence (mean ± std over seeds)")
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            path = os.path.join(save_dir, f"convergence_{obj}_d{dim}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            print(f"Convergence curve saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PSO experiment results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--save-dir", default="results/analysis")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    records = load_results(args.results_dir)
    if not records:
        print("No results found.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loaded {len(records)} results\n")
    print("=== Mean time ± std (seconds) ===")
    print_summary_table(records)
    print_speedup_table(records)

    plot_fitness_boxplots(records, args.save_dir)
    plot_time_bars(records, args.save_dir)

    groups = load_histories(args.results_dir)
    if groups:
        plot_convergence_curves(groups, args.save_dir)


if __name__ == "__main__":
    main()
