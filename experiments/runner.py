import os
import numpy as np

from core.pso import PSO, RunResult
from parallel.sequential import SequentialEvaluator
from parallel.threading_eval import ThreadPoolEvaluator
from parallel.multiprocessing_eval import ProcessPoolEvaluator
from objectives import REGISTRY
from storage.logger import setup_logger
from storage.persistence import result_dir, save_summary_json, save_history_csv


class RunConfig:
    def __init__(
        self,
        objective: str,
        dim: int,
        bounds_lo: float,
        bounds_hi: float,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iters: int = 500,
        tol: float = 1e-8,
        patience: int = 50,
        seed: int = 42,
        evaluator: str = "sequential",
        save_dir: str = "results",
        log_dir: str = "logs",
        compare_baseline: bool = False,
    ):
        self.objective = objective
        self.dim = dim
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iters = max_iters
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.evaluator = evaluator
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.compare_baseline = compare_baseline


def run_experiment(config: RunConfig) -> RunResult:
    obj_info = REGISTRY[config.objective]
    objective = obj_info["fn"]
    lo = np.full(config.dim, config.bounds_lo)
    hi = np.full(config.dim, config.bounds_hi)

    log_name = f"{config.objective}_d{config.dim}_s{config.seed}"
    logger = setup_logger(log_name, log_dir=config.log_dir)

    if config.evaluator == "threading":
        evaluator = ThreadPoolEvaluator(objective)
    elif config.evaluator == "multiprocessing":
        evaluator = ProcessPoolEvaluator(objective)
    else:
        evaluator = SequentialEvaluator(objective)

    pso_runner = PSO(
        evaluator=evaluator,
        bounds_lo=lo,
        bounds_hi=hi,
        n_particles=config.n_particles,
        w=config.w,
        c1=config.c1,
        c2=config.c2,
        max_iters=config.max_iters,
        tol=config.tol,
        patience=config.patience,
        seed=config.seed,
        logger=logger,
    )

    result = pso_runner.run()

    baseline_result = None
    if config.compare_baseline:
        from parallel.baseline import run_pyswarm
        baseline_result = run_pyswarm(
            objective=objective,
            bounds_lo=lo,
            bounds_hi=hi,
            n_particles=config.n_particles,
            w=config.w,
            c1=config.c1,
            c2=config.c2,
            max_iters=config.max_iters,
        )

    if config.save_dir:
        rdir = result_dir(config.save_dir, config.objective, config.dim, config.seed)
        cfg_dict = {
            "objective": config.objective,
            "dim": config.dim,
            "bounds_lo": config.bounds_lo,
            "bounds_hi": config.bounds_hi,
            "n_particles": config.n_particles,
            "w": config.w,
            "c1": config.c1,
            "c2": config.c2,
            "max_iters": config.max_iters,
            "tol": config.tol,
            "patience": config.patience,
            "seed": config.seed,
            "evaluator": config.evaluator,
        }
        res_dict = {
            "best_fitness": result.best_fitness,
            "best_position": result.best_position.tolist(),
            "iterations": result.iterations,
            "time_total": result.time_total,
            "time_eval": result.time_eval,
            "time_update": result.time_update,
        }
        if baseline_result is not None:
            res_dict["baseline_best_fitness"] = baseline_result.best_fitness
            res_dict["baseline_time_total"] = baseline_result.time_total
        save_summary_json(os.path.join(rdir, "summary.json"), cfg_dict, res_dict)
        save_history_csv(os.path.join(rdir, "history.csv"), result.history)

    if baseline_result is not None:
        result._baseline = baseline_result

    return result
