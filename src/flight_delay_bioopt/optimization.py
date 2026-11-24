"""Fair, budgeted hyperparameter-optimization framework for network XGBoost."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.optimize import Bounds, minimize
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from sko.GA import GA
from sko.PSO import PSO

from .data import load_config
from .network_experiment import build_network_preprocessor, prepare_network_splits
from .network_features import build_network_dataset


PARAMETER_ORDER = [
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "scale_pos_weight",
]


@dataclass
class OptimizationMatrices:
    """Only train and validation matrices are exposed to optimizers."""

    x_train: sparse.csr_matrix
    y_train: np.ndarray
    x_validation: sparse.csr_matrix
    y_validation: np.ndarray


def decode_vector(vector: np.ndarray, search_space: dict[str, Any]) -> dict[str, Any]:
    """Map a normalized [0, 1]^5 vector into the shared parameter space."""
    clipped = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    params: dict[str, Any] = {}
    for index, name in enumerate(PARAMETER_ORDER):
        specification = search_space[name]
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        value = clipped[index]
        if specification["type"] == "integer":
            decoded: Any = int(round(minimum + value * (maximum - minimum)))
        elif specification["type"] == "log_float":
            decoded = float(math.exp(math.log(minimum) + value * (math.log(maximum) - math.log(minimum))))
        else:
            decoded = float(minimum + value * (maximum - minimum))
        params[name] = decoded
    return params


def normalize_params(params: dict[str, Any]) -> tuple[Any, ...]:
    """Create a stable cache key without hiding materially different candidates."""
    return tuple(
        int(params[name]) if name == "max_depth" else round(float(params[name]), 8)
        for name in PARAMETER_ORDER
    )


def grid_vectors() -> list[np.ndarray]:
    """Return the complete 2^5 coarse grid used by the full Grid Search."""
    return [
        np.asarray(values, dtype=float)
        for values in __import__("itertools").product([0.0, 1.0], repeat=len(PARAMETER_ORDER))
    ]


def stratified_subsample(
    x: sparse.csr_matrix, y: np.ndarray, size: int, seed: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Take a deterministic class-stratified smoke-test sample."""
    if size >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        indices = np.flatnonzero(y == label)
        label_size = max(1, round(size * len(indices) / len(y)))
        selected.extend(rng.choice(indices, size=min(label_size, len(indices)), replace=False))
    selected_array = np.asarray(sorted(selected), dtype=int)
    return x[selected_array], y[selected_array]


def build_optimization_cache(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Fit preprocessing on train only and cache train/validation matrices."""
    config = load_config(config_path)
    network_path = Path(config["data"]["network_dataset"])
    if not network_path.exists():
        build_network_dataset(config_path)
    splits, network_features, _ = prepare_network_splits(config)
    categorical = config["baseline"]["categorical_features"]
    numeric = config["baseline"]["numeric_features"] + network_features
    features = categorical + numeric
    preprocessor = build_network_preprocessor(config, network_features)
    x_train = sparse.csr_matrix(preprocessor.fit_transform(splits["train"][features]))
    x_validation = sparse.csr_matrix(preprocessor.transform(splits["validation"][features]))
    y_train = splits["train"]["is_delayed"].astype("int8").to_numpy()
    y_validation = splits["validation"]["is_delayed"].astype("int8").to_numpy()

    cache_dir = Path(config["outputs"]["optimization_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache_dir / "x_train.npz", x_train)
    sparse.save_npz(cache_dir / "x_validation.npz", x_validation)
    np.save(cache_dir / "y_train.npy", y_train, allow_pickle=False)
    np.save(cache_dir / "y_validation.npy", y_validation, allow_pickle=False)
    joblib.dump(preprocessor, cache_dir / "preprocessor.joblib")
    metadata = {
        "train_records": int(len(y_train)),
        "validation_records": int(len(y_validation)),
        "matrix_columns": int(x_train.shape[1]),
        "network_features": len(network_features),
        "features": features,
        "test_loaded_by_optimizer": False,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_optimization_matrices(config: dict[str, Any]) -> OptimizationMatrices:
    cache_dir = Path(config["outputs"]["optimization_cache_dir"])
    required = ["x_train.npz", "x_validation.npz", "y_train.npy", "y_validation.npy"]
    if not all((cache_dir / name).exists() for name in required):
        raise FileNotFoundError("Optimization cache is incomplete; build it first")
    return OptimizationMatrices(
        x_train=sparse.load_npz(cache_dir / "x_train.npz").tocsr(),
        y_train=np.load(cache_dir / "y_train.npy", allow_pickle=False),
        x_validation=sparse.load_npz(cache_dir / "x_validation.npz").tocsr(),
        y_validation=np.load(cache_dir / "y_validation.npy", allow_pickle=False),
    )


class BudgetedObjective:
    """Validation-F1 objective with a strict call budget and candidate cache."""

    def __init__(
        self,
        matrices: OptimizationMatrices,
        search_space: dict[str, Any],
        fixed_parameters: dict[str, Any],
        budget: int,
        seed: int,
    ) -> None:
        self.matrices = matrices
        self.search_space = search_space
        self.fixed_parameters = fixed_parameters
        self.budget = budget
        self.seed = seed
        self.calls = 0
        self.model_fits = 0
        self.cache_hits = 0
        self.cache: dict[tuple[Any, ...], float] = {}
        self.history: list[dict[str, Any]] = []
        self.best_score = -1.0
        self.best_params: dict[str, Any] | None = None

    def __call__(self, vector: np.ndarray) -> float:
        if self.calls >= self.budget:
            return self.best_score if self.best_score >= 0 else 0.0
        self.calls += 1
        params = decode_vector(np.asarray(vector).reshape(-1), self.search_space)
        key = normalize_params(params)
        started = perf_counter()
        if key in self.cache:
            score = self.cache[key]
            self.cache_hits += 1
            fitted = False
        else:
            model = XGBClassifier(
                **self.fixed_parameters,
                **params,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.seed,
            )
            model.fit(self.matrices.x_train, self.matrices.y_train)
            prediction = (model.predict_proba(self.matrices.x_validation)[:, 1] >= 0.5).astype("int8")
            score = float(f1_score(self.matrices.y_validation, prediction, zero_division=0))
            self.cache[key] = score
            self.model_fits += 1
            fitted = True
        elapsed = perf_counter() - started
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        self.history.append(
            {
                "evaluation": self.calls,
                **params,
                "validation_f1": score,
                "best_f1_so_far": self.best_score,
                "model_fitted": fitted,
                "seconds": elapsed,
            }
        )
        return score


def _run_grid(objective: BudgetedObjective) -> None:
    for vector in grid_vectors():
        if objective.calls >= objective.budget:
            break
        objective(vector)


def _run_random(objective: BudgetedObjective, seed: int) -> None:
    rng = np.random.default_rng(seed)
    while objective.calls < objective.budget:
        objective(rng.random(len(PARAMETER_ORDER)))


def _run_hill_climbing(objective: BudgetedObjective, seed: int) -> None:
    rng = np.random.default_rng(seed)
    current = np.full(len(PARAMETER_ORDER), 0.5)
    current_score = objective(current)
    step = 0.25
    while objective.calls < objective.budget:
        candidate = current.copy()
        dimension = int(rng.integers(len(candidate)))
        candidate[dimension] = np.clip(
            candidate[dimension] + rng.choice([-step, step]), 0, 1
        )
        score = objective(candidate)
        if score >= current_score:
            current, current_score = candidate, score
        else:
            step = max(0.05, step * 0.9)


def _run_tabu(objective: BudgetedObjective) -> None:
    current = np.full(len(PARAMETER_ORDER), 0.5)
    objective(current)
    tabu: deque[tuple[int, int]] = deque(maxlen=4)
    step = 0.25
    while objective.calls < objective.budget:
        candidates = []
        for dimension in range(len(current)):
            for direction in (-1, 1):
                move = (dimension, direction)
                if move in tabu:
                    continue
                candidate = current.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step, 0, 1
                )
                candidates.append((move, candidate))
        best_move = None
        best_candidate = None
        best_score = -1.0
        for move, candidate in candidates:
            if objective.calls >= objective.budget:
                break
            score = objective(candidate)
            if score > best_score:
                best_move, best_candidate, best_score = move, candidate, score
        if best_candidate is None:
            break
        current = best_candidate
        tabu.append(best_move)  # type: ignore[arg-type]


def _run_simulated_annealing(objective: BudgetedObjective, seed: int) -> None:
    rng = np.random.default_rng(seed)
    current = np.full(len(PARAMETER_ORDER), 0.5)
    current_score = objective(current)
    while objective.calls < objective.budget:
        temperature = max(0.01, 1.0 - objective.calls / objective.budget)
        candidate = np.clip(current + rng.normal(0, 0.2 * temperature, len(current)), 0, 1)
        score = objective(candidate)
        accept = score >= current_score or rng.random() < math.exp(
            (score - current_score) / temperature
        )
        if accept:
            current, current_score = candidate, score


def _run_ga(objective: BudgetedObjective, budget: int, seed: int) -> None:
    np.random.seed(seed)
    population = 4
    iterations = max(1, math.ceil(budget / population))
    optimizer = GA(
        func=lambda vector: -objective(vector),
        n_dim=len(PARAMETER_ORDER),
        size_pop=population,
        max_iter=iterations,
        lb=[0] * len(PARAMETER_ORDER),
        ub=[1] * len(PARAMETER_ORDER),
        precision=1e-3,
        prob_mut=0.1,
    )
    optimizer.run()


def _run_pso(objective: BudgetedObjective, budget: int, seed: int) -> None:
    np.random.seed(seed)
    population = 4
    iterations = max(1, math.ceil(budget / population))
    optimizer = PSO(
        func=lambda vector: -objective(vector),
        n_dim=len(PARAMETER_ORDER),
        pop=population,
        max_iter=iterations,
        lb=[0] * len(PARAMETER_ORDER),
        ub=[1] * len(PARAMETER_ORDER),
        w=0.8,
        c1=0.5,
        c2=0.5,
    )
    optimizer.run()


def _run_scipy_nelder_mead(
    objective: BudgetedObjective, budget: int, seed: int
) -> None:
    """Run bounded SciPy Nelder-Mead under the shared evaluation budget.

    Nelder-Mead is deterministic from its starting simplex, so ``seed`` is
    accepted only to keep the common optimizer signature.  SciPy minimizes;
    the project objective maximizes validation F1, hence the minus sign.
    """
    del seed
    dimensions = len(PARAMETER_ORDER)
    minimize(
        fun=lambda vector: -objective(np.clip(vector, 0.0, 1.0)),
        x0=np.full(dimensions, 0.5),
        method="Nelder-Mead",
        bounds=Bounds(np.zeros(dimensions), np.ones(dimensions)),
        options={
            "adaptive": True,
            "maxfev": budget,
            "maxiter": budget * 4,
            "xatol": 1e-3,
            "fatol": 1e-5,
        },
    )


def _run_bayesian(objective: BudgetedObjective, budget: int, seed: int) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )

    def optuna_objective(trial: optuna.Trial) -> float:
        vector = np.asarray(
            [trial.suggest_float(name, 0.0, 1.0) for name in PARAMETER_ORDER]
        )
        return objective(vector)

    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)


ALGORITHMS: dict[str, Callable[..., None]] = {
    "grid_search": lambda objective, budget, seed: _run_grid(objective),
    "random_search": lambda objective, budget, seed: _run_random(objective, seed),
    "hill_climbing": lambda objective, budget, seed: _run_hill_climbing(objective, seed),
    "tabu_search": lambda objective, budget, seed: _run_tabu(objective),
    "genetic_algorithm": lambda objective, budget, seed: _run_ga(objective, budget, seed),
    "particle_swarm": lambda objective, budget, seed: _run_pso(objective, budget, seed),
    "scipy_nelder_mead": lambda objective, budget, seed: _run_scipy_nelder_mead(
        objective, budget, seed
    ),
    "simulated_annealing": lambda objective, budget, seed: _run_simulated_annealing(objective, seed),
    "bayesian_tpe": lambda objective, budget, seed: _run_bayesian(objective, budget, seed),
}


def run_optimization_smoke_test(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Run a cheap framework check; these scores are not final experiment results."""
    config = load_config(config_path)
    cache_metadata = build_optimization_cache(config_path)
    matrices = load_optimization_matrices(config)
    smoke = config["optimization"]["smoke"]
    seed = config["project"]["random_seed"]
    x_train, y_train = stratified_subsample(
        matrices.x_train, matrices.y_train, smoke["train_records"], seed
    )
    x_validation, y_validation = stratified_subsample(
        matrices.x_validation,
        matrices.y_validation,
        smoke["validation_records"],
        seed + 1,
    )
    smoke_matrices = OptimizationMatrices(x_train, y_train, x_validation, y_validation)
    fixed = dict(config["optimization"]["fixed_parameters"])
    fixed["n_estimators"] = smoke["n_estimators"]
    budget = int(smoke["budget_per_algorithm"])
    output_dir = Path(config["outputs"]["optimization_smoke_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for offset, (name, runner) in enumerate(ALGORITHMS.items()):
        objective = BudgetedObjective(
            smoke_matrices,
            config["optimization"]["search_space"],
            fixed,
            budget,
            seed + offset,
        )
        started = perf_counter()
        runner(objective, budget, seed + offset)
        elapsed = perf_counter() - started
        pd.DataFrame(objective.history).to_csv(output_dir / f"{name}_history.csv", index=False)
        results[name] = {
            "status": "passed" if objective.calls == budget else "incomplete",
            "objective_calls": objective.calls,
            "model_fits": objective.model_fits,
            "cache_hits": objective.cache_hits,
            "best_validation_f1_smoke": objective.best_score,
            "best_params_smoke": objective.best_params,
            "seconds": round(elapsed, 3),
        }

    # One full-data center-point fit gives a practical runtime estimate without
    # starting any full optimization algorithm.
    benchmark_objective = BudgetedObjective(
        matrices,
        config["optimization"]["search_space"],
        config["optimization"]["fixed_parameters"],
        budget=1,
        seed=seed,
    )
    benchmark_objective(np.full(len(PARAMETER_ORDER), 0.5))
    benchmark_seconds = benchmark_objective.history[0]["seconds"]
    full_budget = int(config["optimization"]["full_budget_per_algorithm"])
    estimated_total_minutes = (
        benchmark_seconds * full_budget * len(ALGORITHMS) / 60.0
    )
    summary = {
        "kind": "smoke_test_not_final_results",
        "objective": "validation_f1_at_threshold_0.5",
        "test_loaded_by_optimizer": False,
        "smoke_train_records": int(len(y_train)),
        "smoke_validation_records": int(len(y_validation)),
        "budget_per_algorithm": budget,
        "algorithms": results,
        "full_cache": cache_metadata,
        "full_single_evaluation_seconds": round(float(benchmark_seconds), 3),
        "planned_full_budget_per_algorithm": full_budget,
        "rough_full_runtime_minutes_sequential": round(float(estimated_total_minutes), 1),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [{"algorithm": name, **values} for name, values in results.items()]
    ).drop(columns=["best_params_smoke"]).to_csv(
        output_dir / "smoke_status.csv", index=False
    )
    return summary


def _write_full_progress(output_dir: Path, results: dict[str, Any]) -> None:
    """Persist restartable progress after every completed optimizer."""
    completed = {
        "kind": "full_optimization_validation_only",
        "test_loaded_by_optimizer": False,
        "completed_algorithms": list(results),
        "algorithms": results,
    }
    (output_dir / "progress.json").write_text(
        json.dumps(completed, indent=2), encoding="utf-8"
    )
    rows = [
        {
            "algorithm": name,
            "best_validation_f1": values["best_validation_f1"],
            "objective_calls": values["objective_calls"],
            "model_fits": values["model_fits"],
            "cache_hits": values["cache_hits"],
            "seconds": values["seconds"],
        }
        for name, values in results.items()
    ]
    pd.DataFrame(rows).sort_values("best_validation_f1", ascending=False).to_csv(
        output_dir / "comparison_partial.csv", index=False
    )


def _plot_full_results(output_dir: Path, results: dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid")
    comparison = pd.DataFrame(
        [
            {
                "algorithm": name,
                "best_validation_f1": values["best_validation_f1"],
                "objective_calls": values["objective_calls"],
                "model_fits": values["model_fits"],
                "cache_hits": values["cache_hits"],
                "seconds": values["seconds"],
                "minutes": values["seconds"] / 60.0,
            }
            for name, values in results.items()
        ]
    ).sort_values("best_validation_f1", ascending=False)
    comparison.drop(columns=["minutes"]).to_csv(
        output_dir / "comparison.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=comparison, x="algorithm", y="best_validation_f1", ax=ax)
    ax.set(title="Best validation F1 by optimizer", xlabel="Algorithm", ylabel="Validation F1")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "best_validation_f1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in results:
        history = pd.read_csv(output_dir / f"{name}_history.csv")
        ax.plot(history["evaluation"], history["best_f1_so_far"], label=name)
    ax.set(
        title="Optimizer convergence",
        xlabel="Objective calls",
        ylabel="Best validation F1 so far",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=160)
    plt.close(fig)


def run_full_optimization(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Run or resume all optimizers on train/validation only."""
    config = load_config(config_path)
    cache_dir = Path(config["outputs"]["optimization_cache_dir"])
    if not (cache_dir / "metadata.json").exists():
        build_optimization_cache(config_path)
    matrices = load_optimization_matrices(config)
    budget = int(config["optimization"]["full_budget_per_algorithm"])
    seed = int(config["project"]["random_seed"])
    output_dir = Path(config["outputs"]["optimization_full_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for offset, (name, runner) in enumerate(ALGORITHMS.items()):
        result_path = output_dir / f"{name}_result.json"
        if result_path.exists():
            results[name] = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"SKIP {name}: existing completed result", flush=True)
            continue

        print(f"START {name} ({budget} validation evaluations)", flush=True)
        objective = BudgetedObjective(
            matrices,
            config["optimization"]["search_space"],
            config["optimization"]["fixed_parameters"],
            budget,
            seed + offset,
        )
        started = perf_counter()
        runner(objective, budget, seed + offset)
        elapsed = perf_counter() - started
        history = pd.DataFrame(objective.history)
        history.to_csv(output_dir / f"{name}_history.csv", index=False)
        result = {
            "status": "complete" if objective.calls == budget else "incomplete",
            "objective_calls": objective.calls,
            "model_fits": objective.model_fits,
            "cache_hits": objective.cache_hits,
            "best_validation_f1": objective.best_score,
            "best_params": objective.best_params,
            "seconds": round(elapsed, 3),
            "test_used": False,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[name] = result
        _write_full_progress(output_dir, results)
        print(
            f"DONE {name}: best_validation_f1={objective.best_score:.6f}, "
            f"fits={objective.model_fits}, seconds={elapsed:.1f}",
            flush=True,
        )

    winner_name, winner = max(
        results.items(), key=lambda item: item[1]["best_validation_f1"]
    )
    summary = {
        "kind": "full_optimization_validation_only",
        "objective": "validation_f1_at_threshold_0.5",
        "test_loaded_by_optimizer": False,
        "budget_per_algorithm": budget,
        "winner_selected_by": "validation_f1",
        "winner": winner_name,
        "winner_validation_f1": winner["best_validation_f1"],
        "winner_params": winner["best_params"],
        "algorithms": results,
        "total_seconds": round(sum(values["seconds"] for values in results.values()), 3),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_full_progress(output_dir, results)
    _plot_full_results(output_dir, results)
    return summary
