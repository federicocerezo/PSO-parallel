from itertools import product
from typing import Any, Dict, List

import numpy as np

from objectives import REGISTRY
from experiments.runner import RunConfig, run_experiment


def grid_search(
    objective: str,
    dim: int,
    w_values: List[float],
    c1_values: List[float],
    c2_values: List[float],
    n_particles_values: List[int],
    max_iters: int = 500,
    seeds: List[int] = None,
) -> List[Dict[str, Any]]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    obj_info = REGISTRY[objective]
    combos = list(product(w_values, c1_values, c2_values, n_particles_values))

    results = []
    total = len(combos) * len(seeds)
    done = 0

    for w, c1, c2, n_particles in combos:
        fitness_list = []
        time_list = []

        for seed in seeds:
            rc = RunConfig(
                objective=objective,
                dim=dim,
                bounds_lo=obj_info["bounds"][0],
                bounds_hi=obj_info["bounds"][1],
                n_particles=n_particles,
                w=w,
                c1=c1,
                c2=c2,
                max_iters=max_iters,
                seed=seed,
                save_dir="",
            )
            result = run_experiment(rc)
            fitness_list.append(result.best_fitness)
            time_list.append(result.time_total)
            done += 1
            print(f"  [{done}/{total}] w={w} c1={c1} c2={c2} n={n_particles} s={seed} -> {result.best_fitness:.4e}")

        results.append({
            "w": w,
            "c1": c1,
            "c2": c2,
            "n_particles": n_particles,
            "mean_fitness": float(np.mean(fitness_list)),
            "std_fitness": float(np.std(fitness_list)),
            "mean_time": float(np.mean(time_list)),
        })

    results.sort(key=lambda r: r["mean_fitness"])
    return results
