"""Build leakage-safe network features and run the controlled XGBoost test."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.network_experiment import run_network_xgboost_experiment  # noqa: E402


if __name__ == "__main__":
    result = run_network_xgboost_experiment(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
