import csv
import json
import os
import platform
import subprocess
from typing import Any, Dict, List


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _hardware_info() -> Dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
    }


def result_dir(base_dir: str, objective: str, dim: int, seed: int, evaluator: str = "sequential") -> str:
    name = f"{objective}_d{dim}_s{seed}_{evaluator}"
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_summary_json(
    path: str,
    config: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    summary = {
        "config": config,
        "result": result,
        "git_hash": _git_hash(),
        "hardware": _hardware_info(),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def save_history_csv(path: str, history: List[float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_fitness"])
        for i, val in enumerate(history):
            writer.writerow([i, val])


def load_summary_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_history_csv(path: str) -> List[tuple]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["iteration"]), float(row["best_fitness"])))
    return rows
