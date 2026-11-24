"""Run the fixed Random Forest and XGBoost baselines."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.tree_baselines import run_tree_baselines  # noqa: E402


if __name__ == "__main__":
    result = run_tree_baselines(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
