"""Chronological preprocessing and the first Logistic Regression baseline."""

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
from sklearn.linear_model import LogisticRegression
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

from .data import load_config


SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def add_destination_local_features(
    frame: pd.DataFrame, timezone_mapping: dict[str, str]
) -> pd.DataFrame:
    """Derive local calendar features without storing mixed-timezone datetimes."""
    result = frame.copy()
    result["local_year"] = pd.Series(pd.NA, index=result.index, dtype="Int16")
    result["local_month"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result["local_day_of_week"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result["local_hour"] = pd.Series(pd.NA, index=result.index, dtype="Int8")

    unknown = sorted(set(result["destination"].dropna()) - set(timezone_mapping))
    if unknown:
        raise ValueError(f"Missing destination timezone mapping for: {unknown}")

    for destination, timezone_name in timezone_mapping.items():
        mask = result["destination"].eq(destination)
        if not mask.any():
            continue
        local = result.loc[mask, "scheduled_utc"].dt.tz_convert(timezone_name)
        result.loc[mask, "local_year"] = local.dt.year.to_numpy()
        result.loc[mask, "local_month"] = local.dt.month.to_numpy()
        result.loc[mask, "local_day_of_week"] = local.dt.dayofweek.to_numpy()
        result.loc[mask, "local_hour"] = local.dt.hour.to_numpy()

    result["local_is_weekend"] = (
        result["local_day_of_week"].isin([5, 6]).astype("int8")
    )
    result["local_season"] = result["local_month"].map(SEASONS).astype("string")
    return result


def remove_repeated_copies(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the first occurrence of an exact raw-flight duplicate."""
    subset = ["origin", "destination", "scheduled_utc", "delay_minutes"]
    duplicate_copies = frame.duplicated(subset=subset, keep="first")
    return frame.loc[~duplicate_copies].copy(), int(duplicate_copies.sum())


def chronological_split(
    frame: pd.DataFrame, validation_start: str, test_start: str
) -> dict[str, pd.DataFrame]:
    """Split flights by time, never allowing future rows into earlier sets."""
    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)
    if validation_boundary >= test_boundary:
        raise ValueError("validation_start must be earlier than test_start")

    splits = {
        "train": frame.loc[frame["scheduled_utc"] < validation_boundary].copy(),
        "validation": frame.loc[
            frame["scheduled_utc"].between(
                validation_boundary, test_boundary, inclusive="left"
            )
        ].copy(),
        "test": frame.loc[frame["scheduled_utc"] >= test_boundary].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("At least one chronological split is empty")
    return splits


def build_pipeline(
    config: dict[str, Any], class_weight: str | dict[int, float] | None = None
) -> Pipeline:
    """Create a train-only fitted preprocessing and Logistic Regression pipeline."""
    baseline = config["baseline"]
    model_config = baseline["logistic_regression"]
    categorical = baseline["categorical_features"]
    numeric = baseline["numeric_features"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=model_config["minimum_category_frequency"],
                    dtype=np.float32,
                ),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        solver=model_config["solver"],
        max_iter=model_config["max_iter"],
        tol=model_config["tolerance"],
        class_weight=class_weight,
        random_state=config["project"]["random_seed"],
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def prepare_model_splits(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Apply the approved cleaning, local features, and chronological split."""
    source = Path(config["data"]["processed_dir"]) / "balkan_flights.parquet"
    frame = pd.read_parquet(source)
    original_records = len(frame)
    frame, removed_duplicate_copies = remove_repeated_copies(frame)
    frame = add_destination_local_features(frame, config["timezones"])
    frame = frame.loc[~frame["has_missing_core"] & frame["is_delayed"].notna()].copy()
    splits = chronological_split(
        frame,
        validation_start=config["splits"]["validation_start"],
        test_start=config["splits"]["test_start"],
    )
    metadata = {
        "original_records": int(original_records),
        "duplicate_copies_removed": removed_duplicate_copies,
        "extreme_rows_retained": int(frame["is_extreme_delay"].sum()),
    }
    return splits, metadata


def _evaluate(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_true = frame["is_delayed"].astype("int8").to_numpy()
    probability = pipeline.predict_proba(frame[features])[:, 1]
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


def _plot_evaluation(
    evaluations: dict[str, tuple[pd.DataFrame, np.ndarray]],
    output_dir: Path,
    model_label: str,
) -> None:
    sns.set_theme(style="whitegrid")
    for split_name, (predictions, matrix) in evaluations.items():
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(matrix, display_labels=["On time", "Delayed"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"{model_label} – {split_name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for split_name in ["validation", "test"]:
        predictions = evaluations[split_name][0]
        fpr, tpr, _ = roc_curve(
            predictions["is_delayed"].astype(int), predictions["delay_probability"]
        )
        auc = roc_auc_score(
            predictions["is_delayed"].astype(int), predictions["delay_probability"]
        )
        ax.plot(fpr, tpr, label=f"{split_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def _write_coefficients(pipeline: Pipeline, output_dir: Path) -> None:
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["absolute_coefficient"] = table["coefficient"].abs()
    table.sort_values("absolute_coefficient", ascending=False).to_csv(
        output_dir / "coefficients.csv", index=False
    )


def run_logistic_baseline(
    config_path: str | Path = "config/default.yaml",
    *,
    class_weight: str | dict[int, float] | None = None,
    output_key: str = "baseline_dir",
    model_filename: str = "logistic_regression.joblib",
    model_label: str = "Logistic Regression",
) -> dict[str, Any]:
    """Prepare chronological data and train/evaluate the approved baseline."""
    config = load_config(config_path)
    splits, data_metadata = prepare_model_splits(config)

    features = config["baseline"]["categorical_features"] + config["baseline"][
        "numeric_features"
    ]
    threshold = config["baseline"]["logistic_regression"]["threshold"]
    pipeline = build_pipeline(config, class_weight=class_weight)

    started = perf_counter()
    pipeline.fit(splits["train"][features], splits["train"]["is_delayed"].astype("int8"))
    training_seconds = perf_counter() - started

    metrics: dict[str, Any] = {}
    plot_inputs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    output_dir = Path(config["outputs"][output_key])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_frame in splits.items():
        split_metrics, predictions, matrix = _evaluate(
            pipeline, split_frame, features, threshold
        )
        metrics[split_name] = split_metrics
        predictions.to_parquet(output_dir / f"predictions_{split_name}.parquet", index=False)
        plot_inputs[split_name] = (predictions, matrix)

    model = pipeline.named_steps["model"]
    summary = {
        "model": "LogisticRegression",
        "class_weight": class_weight,
        "threshold": threshold,
        "features": features,
        **data_metadata,
        "training_seconds": round(training_seconds, 3),
        "model_iterations": int(model.n_iter_[0]),
        "converged_before_max_iter": bool(model.n_iter_[0] < model.max_iter),
        "splits": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    joblib.dump(pipeline, output_dir / model_filename)
    _write_coefficients(pipeline, output_dir)
    _plot_evaluation(plot_inputs, output_dir, model_label)
    return summary


def write_baseline_comparison(config: dict[str, Any]) -> pd.DataFrame:
    """Compare the untouched standard baseline with the balanced experiment."""
    standard_path = Path(config["outputs"]["baseline_dir"]) / "metrics.json"
    balanced_dir = Path(config["outputs"]["balanced_baseline_dir"])
    balanced_path = balanced_dir / "metrics.json"
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    balanced = json.loads(balanced_path.read_text(encoding="utf-8"))

    rows = []
    for split_name in ["train", "validation", "test"]:
        for label, result in [("standard", standard), ("balanced", balanced)]:
            metrics = result["splits"][split_name]
            rows.append(
                {
                    "model": label,
                    "split": split_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "predicted_positive_percent": metrics[
                        "predicted_positive_percent"
                    ],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(balanced_dir / "comparison.csv", index=False)

    test_comparison = comparison.loc[comparison["split"].eq("test")].melt(
        id_vars=["model", "split"],
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=test_comparison, x="metric", y="score", hue="model", ax=ax)
    ax.set(title="Test-set comparison", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(balanced_dir / "test_metric_comparison.png", dpi=160)
    plt.close(fig)
    return comparison


def run_balanced_logistic_baseline(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Run the approved class-balanced experiment and compare it to baseline."""
    result = run_logistic_baseline(
        config_path,
        class_weight="balanced",
        output_key="balanced_baseline_dir",
        model_filename="logistic_regression_balanced.joblib",
        model_label="Balanced Logistic Regression",
    )
    config = load_config(config_path)
    write_baseline_comparison(config)
    return result
