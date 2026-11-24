"""Fixed-parameter Random Forest and XGBoost baselines."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .baseline import prepare_model_splits
from .data import load_config


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Use exactly the same input features as Logistic Regression."""
    baseline = config["baseline"]
    minimum_frequency = baseline["logistic_regression"][
        "minimum_category_frequency"
    ]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=minimum_frequency,
                    dtype=np.float32,
                ),
                baseline["categorical_features"],
            ),
            ("numeric", StandardScaler(), baseline["numeric_features"]),
        ],
        remainder="drop",
    )


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    settings = config["tree_baselines"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        max_depth=settings["max_depth"],
        min_samples_leaf=settings["min_samples_leaf"],
        max_features=settings["max_features"],
        class_weight=settings["class_weight"],
        n_jobs=-1,
        random_state=config["project"]["random_seed"],
    )


def build_xgboost(config: dict[str, Any], scale_pos_weight: float) -> XGBClassifier:
    settings = config["tree_baselines"]["xgboost"]
    return XGBClassifier(
        n_estimators=settings["n_estimators"],
        max_depth=settings["max_depth"],
        learning_rate=settings["learning_rate"],
        subsample=settings["subsample"],
        colsample_bytree=settings["colsample_bytree"],
        tree_method=settings["tree_method"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=config["project"]["random_seed"],
    )


def evaluate_pipeline(
    pipeline: Pipeline, frame: pd.DataFrame, features: list[str], threshold: float
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_true = frame["is_delayed"].astype("int8").to_numpy()
    # Call the fitted steps directly. XGBoost 2.1.3 predates scikit-learn 1.6's
    # estimator-tag check, although its fit/predict implementation is compatible.
    transformed = pipeline.named_steps["preprocessing"].transform(frame[features])
    probability = pipeline.named_steps["model"].predict_proba(transformed)[:, 1]
    prediction = (probability >= threshold).astype("int8")
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    metrics = {
        "records": int(len(frame)),
        "positive_records": int(y_true.sum()),
        "positive_percent": round(float(y_true.mean() * 100), 4),
        "predicted_positive_percent": round(float(prediction.mean() * 100), 4),
        "accuracy": round(float(accuracy_score(y_true, prediction)), 6),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "confusion_matrix": matrix.tolist(),
    }
    predictions = frame[["origin", "destination", "scheduled_utc", "is_delayed"]].copy()
    predictions["delay_probability"] = probability.astype("float32")
    predictions["prediction"] = prediction
    return metrics, predictions, matrix


def save_model_outputs(
    name: str,
    label: str,
    pipeline: Pipeline,
    split_outputs: dict[str, tuple[dict[str, Any], pd.DataFrame, np.ndarray]],
    output_dir: Path,
) -> None:
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    joblib.dump(model, model_dir / f"{name}_model.joblib")
    if isinstance(model, XGBClassifier):
        # Save the raw Booster too; unlike the sklearn wrapper, it is independent
        # of sklearn's estimator-tag protocol and reloads cleanly across this setup.
        model.get_booster().save_model(model_dir / "xgboost_booster.json")
    else:
        joblib.dump(pipeline, model_dir / f"{name}.joblib")
    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for split_name in ["validation", "test"]:
        predictions = split_outputs[split_name][1]
        fpr, tpr, _ = roc_curve(
            predictions["is_delayed"].astype(int), predictions["delay_probability"]
        )
        auc = roc_auc_score(
            predictions["is_delayed"].astype(int), predictions["delay_probability"]
        )
        ax.plot(fpr, tpr, label=f"{split_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title=f"{label} ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    for split_name, (_, predictions, matrix) in split_outputs.items():
        predictions.to_parquet(model_dir / f"predictions_{split_name}.parquet", index=False)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(matrix, display_labels=["On time", "Delayed"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"{label} – {split_name}")
        fig.tight_layout()
        fig.savefig(model_dir / f"confusion_matrix_{split_name}.png", dpi=160)
        plt.close(fig)


def _write_four_model_comparison(config: dict[str, Any], tree_results: dict[str, Any]) -> None:
    sources = {
        "logistic_standard": json.loads(
            (Path(config["outputs"]["baseline_dir"]) / "metrics.json").read_text(encoding="utf-8")
        ),
        "logistic_balanced": json.loads(
            (Path(config["outputs"]["balanced_baseline_dir"]) / "metrics.json").read_text(encoding="utf-8")
        ),
        "random_forest": tree_results["models"]["random_forest"],
        "xgboost": tree_results["models"]["xgboost"],
    }
    rows = []
    for model_name, result in sources.items():
        for split_name in ["validation", "test"]:
            metrics = result["splits"][split_name]
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "predicted_positive_percent": metrics["predicted_positive_percent"],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison_dir = Path(config["outputs"]["model_comparison_dir"])
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_dir / "comparison.csv", index=False)

    melted = comparison.loc[comparison["split"].eq("test")].melt(
        id_vars=["model", "split"],
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set(title="Fixed-baseline test comparison", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(comparison_dir / "test_metric_comparison.png", dpi=160)
    plt.close(fig)


def run_tree_baselines(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Train the two approved non-optimized tree baselines."""
    config = load_config(config_path)
    splits, data_metadata = prepare_model_splits(config)
    features = config["baseline"]["categorical_features"] + config["baseline"][
        "numeric_features"
    ]
    threshold = config["baseline"]["logistic_regression"]["threshold"]
    train_y = splits["train"]["is_delayed"].astype("int8")
    negative, positive = np.bincount(train_y, minlength=2)
    scale_pos_weight = float(negative / positive)

    estimators = {
        "random_forest": ("Random Forest", build_random_forest(config)),
        "xgboost": ("XGBoost", build_xgboost(config, scale_pos_weight)),
    }
    output_dir = Path(config["outputs"]["tree_baseline_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        **data_metadata,
        "threshold": threshold,
        "xgboost_scale_pos_weight": round(scale_pos_weight, 6),
        "models": {},
    }

    for name, (label, estimator) in estimators.items():
        pipeline = Pipeline(
            [("preprocessing", build_preprocessor(config)), ("model", estimator)]
        )
        started = perf_counter()
        pipeline.fit(splits["train"][features], train_y)
        training_seconds = perf_counter() - started

        split_outputs = {
            split_name: evaluate_pipeline(pipeline, split_frame, features, threshold)
            for split_name, split_frame in splits.items()
        }
        result = {
            "model": label,
            "training_seconds": round(training_seconds, 3),
            "splits": {
                split_name: values[0] for split_name, values in split_outputs.items()
            },
        }
        results["models"][name] = result
        save_model_outputs(name, label, pipeline, split_outputs, output_dir)

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _write_four_model_comparison(config, results)
    return results
