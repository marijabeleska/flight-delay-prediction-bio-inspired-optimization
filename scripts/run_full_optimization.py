"""Run/resume full validation-only optimization, then evaluate the winner once."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.final_evaluation import evaluate_selected_winner_once  # noqa: E402
from flight_delay_bioopt.optimization import run_full_optimization  # noqa: E402


if __name__ == "__main__":
    optimization = run_full_optimization(ROOT / "config" / "default.yaml")
    print("VALIDATION_OPTIMIZATION_COMPLETE", flush=True)
    print(json.dumps(optimization, indent=2), flush=True)
    final = evaluate_selected_winner_once(ROOT / "config" / "default.yaml")
    print("FINAL_TEST_EVALUATION_COMPLETE", flush=True)
    print(json.dumps(final, indent=2), flush=True)
