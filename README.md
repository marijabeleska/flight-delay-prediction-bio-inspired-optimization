# Flight Delay Prediction with Bio-Inspired Optimization

The project predicts whether a flight to one of ten Balkan airports will arrive at least 15 minutes late. It combines classical machine-learning models, delay-propagation features from the European air-transport network, and hyperparameter optimization methods inspired by natural processes.

The main presentation notebook is [`notebooks/01_project_overview.ipynb`](notebooks/01_project_overview.ipynb). It contains the complete workflow, visualizations, experimental results, and conclusions in English.

## What the project does

1. Loads and cleans historical flight records from 2015–2023.
2. Creates the binary target `is_delayed`, where `1` means an arrival delay of at least 15 minutes.
3. Uses chronological data splits: 2015–2021 for training, 2022 for validation, and 2023 for the final test.
4. Compares Logistic Regression, Random Forest, and XGBoost baselines.
5. Adds lagged network features that describe recent delay and flight activity at European and Balkan airports.
6. Optimizes five XGBoost hyperparameters with an equal budget of 32 validation evaluations per algorithm.
7. Refits the validation-selected configuration and evaluates it once on the held-out 2023 test set.

## Optimization algorithms

The project compares nine search methods:

- Grid Search (GS)
- Random Search (RS)
- Hill Climbing (HC)
- Tabu Search (TS)
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Bayesian optimization with TPE
- SciPy Nelder–Mead (SOpt)

Additional runs with seeds 100 and 101 are included to examine the stability of the randomized optimizers. Across the three completed runs, Bayesian/TPE achieved the highest mean validation F1, while Tabu Search was almost tied and more stable.

## Project structure

```text
config/       Data paths, feature settings, model settings, and search space
data/         Local raw, interim, and processed datasets (excluded from Git)
notebooks/    Presentation notebook with the complete project workflow
reports/      Written project report
results/      Saved metrics, tables, plots, models, and optimization histories
scripts/      Commands for data preparation, experiments, and summaries
src/          Reusable Python package for data, features, models, and optimization
tests/        Automated tests
```

Data files are not stored in this repository. Raw-data locations are configured in [`config/default.yaml`](config/default.yaml), while datasets generated locally by the pipeline are written under `data/processed/`.

## Setup

The project can use an existing Python or Miniconda installation; a separate Conda environment is not required. Python 3.11 or 3.12 is supported.

From the project root, install the required libraries and the local package, then check the environment:

```powershell
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost optuna pyarrow matplotlib seaborn jupyterlab ipykernel joblib tqdm pyyaml pytest scikit-opt
python -m pip install -e .
python scripts/check_environment.py
```

Before running the pipeline on another computer, update the raw-data paths in `config/default.yaml`.

Open the presentation notebook with:

```powershell
jupyter lab notebooks/01_project_overview.ipynb
```

The optional [`environment.yml`](environment.yml) provides a reproducible Conda specification if an isolated environment is preferred.

## Running experiments

The main scripts are:

```powershell
python scripts/run_data_audit.py
python scripts/run_logistic_baseline.py
python scripts/run_balanced_logistic_baseline.py
python scripts/run_tree_baselines.py
python scripts/run_network_experiment.py
python scripts/run_optimization_smoke.py
python scripts/run_full_optimization.py
```

To inspect and run an additional shared seed without repeating completed algorithm/seed combinations:

```powershell
python scripts/run_additional_seed.py --seed 101 --dry-run
python scripts/run_additional_seed.py --seed 101
```

Completed results are written to `results/optimization_multiseed/seed_<SEED>/`. Rebuild the combined table and mean ± standard-deviation plot without retraining by running:

```powershell
python scripts/summarize_multiseed.py
```

## Tests

```powershell
python -m pytest -q
```

