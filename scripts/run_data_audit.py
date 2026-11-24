"""Run the Balkan flight data audit from the repository root."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.data import run_data_audit  # noqa: E402


if __name__ == "__main__":
    result = run_data_audit(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
