"""Rebuild tables and the mean +/- standard-deviation plot without training."""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay_bioopt.multiseed import write_multiseed_summary  # noqa: E402


if __name__ == "__main__":
    result = write_multiseed_summary(ROOT / "config" / "default.yaml")
    print(json.dumps(result, indent=2))
