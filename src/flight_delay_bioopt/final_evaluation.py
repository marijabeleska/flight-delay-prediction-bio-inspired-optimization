"""One-time test evaluation after validation-only optimizer selection."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from .data import load_config
from .network_experiment import build_network_preprocessor, prepare_network_splits
from .tree_baselines import build_xgboost, evaluate_pipeline


def evaluate_selected_winner_once(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Refit the validation-selected configuration and open test exactly once."""
    config = load_config(config_path)
    output_dir = Path(config["outputs"]["optimization_full_dir"])
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Full validation-only optimization is not complete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(summary["algorithms"]) != 8:
        raise ValueError("All eight optimizers must finish before test evaluation")

    final_path = output_dir / "final_test_metrics.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))

    splits, network_features, _ = prepare_network_splits(config)
    features = (
        config["baseline"]["categorical_features"]
        + config["baseline"]["numeric_features"]
        + network_features
    )
    train_validation = pd.concat(
        [splits["train"], splits["validation"]], ignore_index=True
    ).sort_values("scheduled_utc", kind="stable")
    preprocessor = build_network_preprocessor(config, network_features)
    x_train_validation = preprocessor.fit_transform(train_validation[features])
    y_train_validation = train_validation["is_delayed"].astype("int8")

    fixed = dict(config["optimization"]["fixed_parameters"])
    winner_params = summary["winner_params"]
    model = build_xgboost(config, scale_pos_weight=winner_params["scale_pos_weight"])
    model.set_params(
        n_estimators=fixed["n_estimators"],
        colsample_bytree=fixed["colsample_bytree"],
        tree_method=fixed["tree_method"],
        max_depth=winner_params["max_depth"],
        learning_rate=winner_params["learning_rate"],
        min_child_weight=winner_params["min_child_weight"],
        subsample=winner_params["subsample"],
    )
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])
    started = perf_counter()
    model.fit(x_train_validation, y_train_validation)
    training_seconds = perf_counter() - started

    threshold = config["baseline"]["logistic_regression"]["threshold"]
    test_metrics, predictions, matrix = evaluate_pipeline(
        pipeline, splits["test"], features, threshold
    )
    final_dir = output_dir / "final_winner"
    final_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, final_dir / "preprocessor.joblib")
    joblib.dump(model, final_dir / "xgboost_winner_model.joblib")
    model.get_booster().save_model(final_dir / "xgboost_winner_booster.json")
    predictions.to_parquet(final_dir / "test_predictions.parquet", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["On time", "Delayed"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Final validation-selected winner – test")
    fig.tight_layout()
    fig.savefig(final_dir / "confusion_matrix_test.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(
        predictions["is_delayed"].astype(int), predictions["delay_probability"]
    )
    auc = roc_auc_score(
        predictions["is_delayed"].astype(int), predictions["delay_probability"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Test AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Final winner ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(final_dir / "roc_test.png", dpi=160)
    plt.close(fig)

    result = {
        "selection_used": "validation_f1_only",
        "selected_optimizer": summary["winner"],
        "selected_validation_f1": summary["winner_validation_f1"],
        "selected_params": winner_params,
        "refit_records": int(len(train_validation)),
        "refit_period": "train_plus_validation_2015_2022",
        "training_seconds": round(training_seconds, 3),
        "test_evaluated_once": True,
        "test": test_metrics,
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
