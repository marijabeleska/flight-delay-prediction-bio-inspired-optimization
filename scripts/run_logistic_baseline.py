"""Run the approved chronological Logistic Regression baseline."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.baseline import run_logistic_baseline  # noqa: E402


if __name__ == "__main__":
    result = run_logistic_baseline(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
