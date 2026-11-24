"""Controlled XGBoost comparison with and without network-delay features."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .baseline import (
    add_destination_local_features,
    chronological_split,
    remove_repeated_copies,
)
from .data import load_config
from .network_features import build_network_dataset
from .tree_baselines import build_xgboost, evaluate_pipeline, save_model_outputs


def prepare_network_splits(
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, int]]:
    """Load network data and apply exactly the approved baseline cleaning/splits."""
    frame = pd.read_parquet(config["data"]["network_dataset"])
    original_records = len(frame)
    frame, removed = remove_repeated_copies(frame)
    frame = add_destination_local_features(frame, config["timezones"])
    frame = frame.loc[
        ~frame["has_missing_core"]
        & frame["is_delayed"].notna()
        & frame["network_feature_coverage"]
    ].copy()
    network_features = sorted(
        column for column in frame.columns if column.startswith("network_")
        and column not in {"network_anchor_utc", "network_feature_coverage"}
    )
    splits = chronological_split(
        frame,
        validation_start=config["splits"]["validation_start"],
        test_start=config["splits"]["test_start"],
    )
    metadata = {
        "original_records": int(original_records),
        "duplicate_copies_removed": removed,
        "covered_records_after_cleaning": int(len(frame)),
        "extreme_rows_retained": int(frame["is_extreme_delay"].sum()),
    }
    return splits, network_features, metadata


def build_network_preprocessor(
    config: dict[str, Any], network_features: list[str]
) -> ColumnTransformer:
    baseline = config["baseline"]
    minimum_frequency = baseline["logistic_regression"]["minimum_category_frequency"]
    numeric = baseline["numeric_features"] + network_features
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
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def run_network_xgboost_experiment(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Build network features and run the one controlled XGBoost experiment."""
    config = load_config(config_path)
    dataset_summary = build_network_dataset(config_path)
    splits, network_features, data_metadata = prepare_network_splits(config)
    features = (
        config["baseline"]["categorical_features"]
        + config["baseline"]["numeric_features"]
        + network_features
    )
    y_train = splits["train"]["is_delayed"].astype("int8")
    negative, positive = np.bincount(y_train, minlength=2)
    scale_pos_weight = float(negative / positive)
    model = build_xgboost(config, scale_pos_weight)
    pipeline = Pipeline(
        [
            ("preprocessing", build_network_preprocessor(config, network_features)),
            ("model", model),
        ]
    )

    started = perf_counter()
    pipeline.fit(splits["train"][features], y_train)
    training_seconds = perf_counter() - started
    threshold = config["baseline"]["logistic_regression"]["threshold"]
    split_outputs = {
        name: evaluate_pipeline(pipeline, frame, features, threshold)
        for name, frame in splits.items()
    }

    output_dir = Path(config["outputs"]["network_experiment_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_outputs(
        "xgboost_network",
        "XGBoost with network features",
        pipeline,
        split_outputs,
        output_dir,
    )
    result = {
        "model": "XGBoost with network features",
        "threshold": threshold,
        "scale_pos_weight": round(scale_pos_weight, 6),
        "training_seconds": round(training_seconds, 3),
        "network_feature_count": len(network_features),
        "network_features": network_features,
        "dataset": dataset_summary,
        **data_metadata,
        "splits": {name: values[0] for name, values in split_outputs.items()},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    base = json.loads(
        (Path(config["outputs"]["tree_baseline_dir"]) / "metrics.json").read_text(
            encoding="utf-8"
        )
    )["models"]["xgboost"]
    rows = []
    for label, metrics_source in [("xgboost_base", base), ("xgboost_network", result)]:
        for split_name in ["validation", "test"]:
            metrics = metrics_source["splits"][split_name]
            rows.append(
                {
                    "model": label,
                    "split": split_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "predicted_positive_percent": metrics[
                        "predicted_positive_percent"
                    ],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    test = comparison.loc[comparison["split"].eq("test")].melt(
        id_vars=["model", "split"],
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=test, x="metric", y="score", hue="model", ax=ax)
    ax.set(title="XGBoost network-feature comparison", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "test_metric_comparison.png", dpi=160)
    plt.close(fig)
    return result
