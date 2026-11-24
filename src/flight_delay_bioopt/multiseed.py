"""Incremental, restartable multi-seed optimization experiments."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .data import load_config
from .optimization import ALGORITHMS, BudgetedObjective, load_optimization_matrices


# These values reproduce the order used by the original completed run before
# SciPy/Nelder-Mead was inserted into the algorithm registry.
ORIGINAL_RESULT_SEEDS = {
    "grid_search": 42,
    "random_search": 43,
    "hill_climbing": 44,
    "tabu_search": 45,
    "genetic_algorithm": 46,
    "particle_swarm": 47,
    "simulated_annealing": 48,
    "bayesian_tpe": 49,
    "scipy_nelder_mead": 48,
}


def parse_algorithm_selection(value: str) -> list[str]:
    """Parse ``all`` or a comma-separated list in registry order."""
    if value.strip().lower() == "all":
        return list(ALGORITHMS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(ALGORITHMS))
    if unknown:
        raise ValueError(
            f"Unknown algorithms: {', '.join(unknown)}. "
            f"Available: {', '.join(ALGORITHMS)}"
        )
    requested_set = set(requested)
    return [name for name in ALGORITHMS if name in requested_set]


def _completed_result(path: Path, seed: int, budget: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("status") == "complete"
        and result.get("seed") == seed
        and result.get("objective_calls") == budget
    ):
        return result
    raise ValueError(f"Existing result is incomplete or mismatched: {path}")


def plan_seed_run(
    config_path: str | Path,
    seed: int,
    algorithms: list[str],
) -> list[dict[str, Any]]:
    """Return a no-training plan with skip/run status and a rough time estimate."""
    config = load_config(config_path)
    budget = int(config["optimization"]["full_budget_per_algorithm"])
    multi_dir = Path(config["outputs"]["optimization_full_dir"]).parent / "optimization_multiseed"
    original_dir = Path(config["outputs"]["optimization_full_dir"])
    rows = []
    for name in algorithms:
        result_path = multi_dir / f"seed_{seed}" / f"{name}_result.json"
        status = "skip_complete" if _completed_result(result_path, seed, budget) else "run"
        original_path = original_dir / f"{name}_result.json"
        estimated_seconds = None
        if original_path.exists():
            estimated_seconds = json.loads(original_path.read_text(encoding="utf-8")).get("seconds")
        rows.append(
            {
                "algorithm": name,
                "seed": seed,
                "status": status,
                "estimated_seconds": estimated_seconds,
                "result_path": str(result_path),
            }
        )
    return rows


def run_additional_seed(
    config_path: str | Path,
    seed: int,
    algorithms: list[str],
) -> dict[str, Any]:
    """Run only missing algorithm results for one explicit shared seed."""
    config = load_config(config_path)
    matrices = load_optimization_matrices(config)
    budget = int(config["optimization"]["full_budget_per_algorithm"])
    multi_dir = Path(config["outputs"]["optimization_full_dir"]).parent / "optimization_multiseed"
    seed_dir = multi_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for name in algorithms:
        result_path = seed_dir / f"{name}_result.json"
        completed = _completed_result(result_path, seed, budget)
        if completed is not None:
            results[name] = completed
            print(f"SKIP {name} seed={seed}: complete", flush=True)
            continue

        print(f"START {name} seed={seed} ({budget} validation evaluations)", flush=True)
        objective = BudgetedObjective(
            matrices,
            config["optimization"]["search_space"],
            config["optimization"]["fixed_parameters"],
            budget,
            seed,
        )
        started = perf_counter()
        ALGORITHMS[name](objective, budget, seed)
        elapsed = perf_counter() - started

        history = pd.DataFrame(objective.history)
        history.insert(0, "seed", seed)
        history.insert(0, "algorithm", name)
        history.to_csv(seed_dir / f"{name}_history.csv", index=False)

        result = {
            "kind": "additional_seed_validation_only",
            "status": "complete" if objective.calls == budget else "incomplete",
            "algorithm": name,
            "seed": seed,
            "objective_calls": objective.calls,
            "model_fits": objective.model_fits,
            "cache_hits": objective.cache_hits,
            "best_validation_f1": objective.best_score,
            "best_params": objective.best_params,
            "seconds": round(elapsed, 3),
            "test_used": False,
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(result_path)
        results[name] = result

        progress = {
            "kind": "additional_seed_validation_only",
            "seed": seed,
            "budget_per_algorithm": budget,
            "test_loaded_by_optimizer": False,
            "requested_algorithms": algorithms,
            "completed_algorithms": list(results),
        }
        (seed_dir / "progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        print(
            f"DONE {name} seed={seed}: best_validation_f1={objective.best_score:.6f}, "
            f"fits={objective.model_fits}, seconds={elapsed:.1f}",
            flush=True,
        )

    write_multiseed_summary(config_path)
    return {
        "seed": seed,
        "budget_per_algorithm": budget,
        "completed_algorithms": list(results),
        "output_dir": str(seed_dir),
        "test_loaded_by_optimizer": False,
    }


def _collect_runs(config: dict[str, Any]) -> pd.DataFrame:
    full_dir = Path(config["outputs"]["optimization_full_dir"])
    multi_dir = full_dir.parent / "optimization_multiseed"
    rows: list[dict[str, Any]] = []

    for algorithm, seed in ORIGINAL_RESULT_SEEDS.items():
        path = full_dir / f"{algorithm}_result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "run_source": "original",
                "best_validation_f1": result["best_validation_f1"],
                "seconds": result["seconds"],
                "result_path": str(path),
            }
        )

    if multi_dir.exists():
        for path in sorted(multi_dir.glob("seed_*/*_result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                continue
            rows.append(
                {
                    "algorithm": result["algorithm"],
                    "seed": result["seed"],
                    "run_source": "additional",
                    "best_validation_f1": result["best_validation_f1"],
                    "seconds": result["seconds"],
                    "result_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def write_multiseed_summary(config_path: str | Path) -> dict[str, Any]:
    """Aggregate the original and every completed additional-seed result."""
    config = load_config(config_path)
    output_dir = Path(config["outputs"]["optimization_full_dir"]).parent / "optimization_multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(config)
    if runs.empty:
        raise ValueError("No completed optimization results were found")
    runs.to_csv(output_dir / "runs.csv", index=False)

    summary = (
        runs.groupby("algorithm", as_index=False)
        .agg(
            runs=("best_validation_f1", "size"),
            f1_mean=("best_validation_f1", "mean"),
            f1_std=("best_validation_f1", "std"),
            f1_best=("best_validation_f1", "max"),
            f1_worst=("best_validation_f1", "min"),
            mean_seconds=("seconds", "mean"),
        )
        .sort_values("f1_mean", ascending=False)
    )
    summary["f1_std"] = summary["f1_std"].fillna(0.0)
    summary.to_csv(output_dir / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = summary.sort_values("f1_mean")
    ax.barh(ordered["algorithm"], ordered["f1_mean"], xerr=ordered["f1_std"], capsize=4)
    ax.set(
        title="Validation F1 across completed seeds",
        xlabel="Mean validation F1 (error bar = standard deviation)",
        ylabel="Algorithm",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "mean_f1_with_std.png", dpi=160)
    plt.close(fig)

    payload = {
        "kind": "multiseed_validation_summary",
        "test_used": False,
        "note": (
            "Original runs used algorithm-specific seeds 42..49. Additional runs "
            "use one shared explicit seed per seed directory."
        ),
        "completed_run_count": int(len(runs)),
        "algorithms": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
