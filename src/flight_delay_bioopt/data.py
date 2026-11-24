"""Load the raw flight files and build the first leakage-safe ML table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


RAW_COLUMNS = ["origin", "destination", "scheduled_timestamp", "delay_seconds"]


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Read the project YAML configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _season(month: pd.Series) -> pd.Series:
    mapping = {
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
    return month.map(mapping).astype("category")


def load_airport_file(
    path: str | Path,
    delay_threshold_minutes: float = 15,
    plausible_min_minutes: float = -180,
    plausible_max_minutes: float = 720,
) -> pd.DataFrame:
    """Load one trusted NumPy file and derive audit and baseline features."""
    file_path = Path(path)
    array = np.load(file_path, allow_pickle=False)

    if array.ndim != 2 or array.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected a 2D array with four columns in {file_path}, got {array.shape}."
        )

    frame = pd.DataFrame(array, columns=RAW_COLUMNS)
    frame["origin"] = frame["origin"].astype("string").str.strip().str.upper()
    frame["destination"] = (
        frame["destination"].astype("string").str.strip().str.upper()
    )
    frame["scheduled_timestamp"] = pd.to_numeric(
        frame["scheduled_timestamp"], errors="coerce"
    )
    frame["delay_seconds"] = pd.to_numeric(frame["delay_seconds"], errors="coerce")

    frame["scheduled_utc"] = pd.to_datetime(
        frame["scheduled_timestamp"], unit="s", utc=True, errors="coerce"
    )
    frame["delay_minutes"] = (frame["delay_seconds"] / 60.0).astype("float32")
    frame["source_airport_file"] = file_path.stem.removeprefix("Delays_")

    frame["year"] = frame["scheduled_utc"].dt.year.astype("Int16")
    frame["month"] = frame["scheduled_utc"].dt.month.astype("Int8")
    frame["day_of_week"] = frame["scheduled_utc"].dt.dayofweek.astype("Int8")
    frame["hour_utc"] = frame["scheduled_utc"].dt.hour.astype("Int8")
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype("int8")
    frame["season"] = _season(frame["month"])
    frame["route"] = frame["origin"] + "_" + frame["destination"]

    has_delay = frame["delay_minutes"].notna()
    frame["is_delayed"] = (
        frame["delay_minutes"].ge(delay_threshold_minutes).where(has_delay).astype("Int8")
    )

    blank_origin = frame["origin"].fillna("").eq("")
    blank_destination = frame["destination"].fillna("").eq("")
    frame["has_missing_core"] = (
        blank_origin
        | blank_destination
        | frame["scheduled_utc"].isna()
        | frame["delay_minutes"].isna()
    )
    frame["destination_file_mismatch"] = (
        frame["destination"] != frame["source_airport_file"]
    ).fillna(True)
    frame["is_extreme_delay"] = (
        frame["delay_minutes"].lt(plausible_min_minutes)
        | frame["delay_minutes"].gt(plausible_max_minutes)
    ).fillna(False)
    frame["is_exact_duplicate"] = frame.duplicated(
        subset=["origin", "destination", "scheduled_utc", "delay_minutes"],
        keep=False,
    )

    return frame.drop(columns=["scheduled_timestamp", "delay_seconds"])


def load_target_airports(config: dict[str, Any]) -> pd.DataFrame:
    """Load and concatenate all configured target-airport files."""
    raw_dir = Path(config["data"]["raw_dir"])
    prediction = config["prediction"]
    plausible = prediction["plausible_delay_range_minutes"]
    frames = []

    for airport in prediction["target_airports"]:
        path = raw_dir / f"Delays_{airport}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing configured airport file: {path}")
        frames.append(
            load_airport_file(
                path,
                delay_threshold_minutes=prediction["delay_threshold_minutes"],
                plausible_min_minutes=plausible["minimum"],
                plausible_max_minutes=plausible["maximum"],
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        "scheduled_utc", kind="stable"
    )


def build_airport_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one audit row per destination airport."""
    summary = (
        frame.groupby("destination", observed=True)
        .agg(
            records=("destination", "size"),
            unique_origins=("origin", "nunique"),
            start_date=("scheduled_utc", "min"),
            end_date=("scheduled_utc", "max"),
            mean_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
            std_delay_minutes=("delay_minutes", "std"),
            delayed_rate=("is_delayed", "mean"),
            missing_core_rows=("has_missing_core", "sum"),
            extreme_delay_rows=("is_extreme_delay", "sum"),
            duplicate_rows=("is_exact_duplicate", "sum"),
            destination_mismatches=("destination_file_mismatch", "sum"),
        )
        .reset_index()
    )
    summary["delayed_percent"] = summary.pop("delayed_rate") * 100
    numeric = summary.select_dtypes(include="number").columns
    summary[numeric] = summary[numeric].round(3)
    return summary


def _save_plots(frame: pd.DataFrame, airport_summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = airport_summary.sort_values("delayed_percent", ascending=False)
    sns.barplot(data=ordered, x="destination", y="delayed_percent", ax=ax)
    ax.set(title="Flights delayed at least 15 minutes", xlabel="Destination", ylabel="Delayed flights (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "delayed_rate_by_airport.png", dpi=160)
    plt.close(fig)

    yearly = frame.groupby("year", observed=True).size().rename("records").reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=yearly, x="year", y="records", ax=ax)
    ax.set(title="Available flight records by year", xlabel="Year", ylabel="Records")
    fig.tight_layout()
    fig.savefig(output / "records_by_year.png", dpi=160)
    plt.close(fig)

    clipped = frame.loc[frame["delay_minutes"].between(-60, 180), "delay_minutes"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(clipped, bins=120, ax=ax)
    ax.axvline(15, color="red", linestyle="--", label="15-minute threshold")
    ax.set(title="Delay distribution (view clipped to -60..180 min)", xlabel="Delay (minutes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "delay_distribution.png", dpi=160)
    plt.close(fig)


def write_audit(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Persist the ML table, audit summaries, plots, and a short Markdown report."""
    processed_dir = Path(config["data"]["processed_dir"])
    audit_dir = Path(config["outputs"]["audit_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "balkan_flights.parquet"
    frame.to_parquet(dataset_path, index=False)

    airport_summary = build_airport_summary(frame)
    airport_summary.to_csv(audit_dir / "airport_summary.csv", index=False)

    year_summary = (
        frame.groupby("year", observed=True)
        .agg(records=("year", "size"), delayed_percent=("is_delayed", "mean"))
        .reset_index()
    )
    year_summary["delayed_percent"] = (year_summary["delayed_percent"] * 100).round(3)
    year_summary.to_csv(audit_dir / "year_summary.csv", index=False)

    route_summary = (
        frame.groupby(["origin", "destination", "route"], observed=True)
        .agg(records=("route", "size"), delayed_percent=("is_delayed", "mean"))
        .reset_index()
        .sort_values("records", ascending=False)
    )
    route_summary["delayed_percent"] = (route_summary["delayed_percent"] * 100).round(3)
    route_summary.to_csv(audit_dir / "route_summary.csv", index=False)

    summary = {
        "records": int(len(frame)),
        "airports": int(frame["destination"].nunique()),
        "origins": int(frame["origin"].nunique()),
        "routes": int(frame["route"].nunique()),
        "date_start_utc": frame["scheduled_utc"].min().isoformat(),
        "date_end_utc": frame["scheduled_utc"].max().isoformat(),
        "delayed_records": int(frame["is_delayed"].sum()),
        "delayed_percent": round(float(frame["is_delayed"].mean() * 100), 3),
        "missing_core_rows": int(frame["has_missing_core"].sum()),
        "exact_duplicate_rows": int(frame["is_exact_duplicate"].sum()),
        "destination_mismatch_rows": int(frame["destination_file_mismatch"].sum()),
        "extreme_delay_rows": int(frame["is_extreme_delay"].sum()),
        "delay_minutes": {
            "minimum": float(frame["delay_minutes"].min()),
            "median": float(frame["delay_minutes"].median()),
            "mean": float(frame["delay_minutes"].mean()),
            "maximum": float(frame["delay_minutes"].max()),
        },
        "dataset_path": str(dataset_path),
    }
    (audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = f"""# Data audit: Balkan flight-delay dataset

- Records: **{summary['records']:,}**
- Destination airports: **{summary['airports']}**
- Unique origins: **{summary['origins']}**
- Routes: **{summary['routes']}**
- Period (UTC): **{summary['date_start_utc']} – {summary['date_end_utc']}**
- Delayed at least 15 minutes: **{summary['delayed_percent']:.3f}%**
- Missing core rows: **{summary['missing_core_rows']}**
- Rows participating in an exact duplicate: **{summary['exact_duplicate_rows']}**
- Destination/file mismatches: **{summary['destination_mismatch_rows']}**
- Extreme-delay flags: **{summary['extreme_delay_rows']}**

Extreme-delay rows are flagged for review, not removed. The initial time features use
UTC because a verified local-timezone mapping has not yet been added for every airport.
No ML model is trained by this audit.
"""
    (audit_dir / "report.md").write_text(report, encoding="utf-8")
    _save_plots(frame, airport_summary, audit_dir)
    return summary


def run_data_audit(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Execute the complete approved data-audit phase."""
    config = load_config(config_path)
    frame = load_target_airports(config)
    return write_audit(frame, config)
