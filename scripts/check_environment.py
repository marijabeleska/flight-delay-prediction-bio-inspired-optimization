"""Quick smoke test for the project's core environment."""

from importlib import import_module
from pathlib import Path
import sys

import yaml


PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "XGBoost": "xgboost",
    "Optuna": "optuna",
    "scikit-opt": "sko",
    "PyArrow": "pyarrow",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    failed = []

    for label, module_name in PACKAGES.items():
        try:
            module = import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{label}: {version}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed.append(label)
            print(f"{label}: FAILED ({exc})")

    config_path = Path("config/default.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(config["data"]["raw_dir"])
    raw_files = list(raw_dir.glob("Delays_*.npy")) if raw_dir.exists() else []
    print(f"Raw data directory: {raw_dir}")
    print(f"Raw airport files found: {len(raw_files)}")

    if failed or len(raw_files) != 20:
        print(f"Environment check failed. Missing packages: {failed}")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

