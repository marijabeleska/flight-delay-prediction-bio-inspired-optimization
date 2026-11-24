"""Run or resume validation-only optimization for one additional seed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.multiseed import (  # noqa: E402
    parse_algorithm_selection,
    plan_seed_run,
    run_additional_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run/resume all missing optimizers for one explicit shared seed."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--algorithms",
        default="all",
        help="'all' or comma-separated internal names, e.g. genetic_algorithm,bayesian_tpe",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without training")
    args = parser.parse_args()

    config_path = ROOT / "config" / "default.yaml"
    algorithms = parse_algorithm_selection(args.algorithms)
    plan = plan_seed_run(config_path, args.seed, algorithms)
    print(f"seed={args.seed}  algorithms={len(algorithms)}")
    for row in plan:
        estimate = row["estimated_seconds"]
        estimate_text = "unknown" if estimate is None else f"~{estimate / 60:.1f} min"
        print(f"  {row['algorithm']:<24} {row['status']:<13} {estimate_text}")
    total = sum(
        row["estimated_seconds"] or 0
        for row in plan
        if row["status"] == "run"
    )
    print(f"rough remaining time: {total / 60:.1f} min")
    print("test data loaded by optimizer: no")

    if args.dry_run:
        return
    run_additional_seed(config_path, args.seed, algorithms)


if __name__ == "__main__":
    main()
