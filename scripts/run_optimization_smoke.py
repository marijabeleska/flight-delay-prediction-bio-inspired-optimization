"""Build the optimization cache and run only the approved cheap smoke test."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.optimization import run_optimization_smoke_test  # noqa: E402


if __name__ == "__main__":
    result = run_optimization_smoke_test(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
