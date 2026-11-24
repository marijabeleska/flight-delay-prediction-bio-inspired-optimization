"""Leakage-safe hourly network features for one-hour-ahead delay prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_config


def load_hourly_sources(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hourly delay and activity tables with a verified UTC index."""
    delays = pd.read_csv(config["data"]["hourly_delays"], parse_dates=["timestamp"])
    activity = pd.read_csv(config["data"]["hourly_activity"], parse_dates=["timestamp"])
    delays["timestamp"] = pd.to_datetime(delays["timestamp"], utc=True)
    activity["timestamp"] = pd.to_datetime(activity["timestamp"], utc=True)
    delays = delays.set_index("timestamp").sort_index()
    activity = activity.set_index("timestamp").sort_index()

    if not delays.index.equals(activity.index):
        raise ValueError("Hourly delay and activity timestamps do not match")
    if set(delays.columns) != set(activity.columns):
        raise ValueError("Hourly delay and activity airport columns do not match")
    if delays.index.duplicated().any():
        raise ValueError("Hourly source contains duplicate timestamps")
    return delays.astype("float32"), activity.astype("int8")


def build_hourly_network_features(
    delays: pd.DataFrame,
    activity: pd.DataFrame,
    european_airports: list[str],
    balkan_airports: list[str],
    lags_hours: list[int],
    rolling_windows_hours: list[int],
) -> pd.DataFrame:
    """Build features whose row at h is safe for flights in the following hour."""
    expected = set(european_airports) | set(balkan_airports)
    missing = sorted(expected - set(delays.columns))
    if missing:
        raise ValueError(f"Missing configured network airports: {missing}")

    active_delays = delays.where(activity.astype(bool))
    feature_data: dict[str, pd.Series] = {}

    # A row indexed by h is joined to flights in h+1. Therefore shift(0) is
    # explicitly named lag_1h relative to the flight prediction time.
    for airport in sorted(expected):
        for lag in lags_hours:
            shift = lag - 1
            feature_data[f"network_delay_{airport}_lag_{lag}h"] = active_delays[
                airport
            ].shift(shift)
            feature_data[f"network_active_{airport}_lag_{lag}h"] = activity[airport].shift(
                shift
            )

    for region, airports in {
        "europe": european_airports,
        "balkans": balkan_airports,
        "all": sorted(expected),
    }.items():
        region_delays = active_delays[airports]
        region_activity = activity[airports]
        mean_delay = region_delays.mean(axis=1)

        feature_data[f"network_{region}_mean_delay_lag_1h"] = mean_delay
        feature_data[f"network_{region}_max_delay_lag_1h"] = region_delays.max(axis=1)
        feature_data[f"network_{region}_active_count_lag_1h"] = region_activity.sum(axis=1)
        feature_data[f"network_{region}_delayed_count_lag_1h"] = (
            region_delays.ge(15).sum(axis=1)
        )

        for window in rolling_windows_hours:
            feature_data[f"network_{region}_mean_delay_rolling_{window}h"] = mean_delay.rolling(
                window=window, min_periods=1
            ).mean()
            feature_data[f"network_{region}_active_count_rolling_{window}h"] = (
                region_activity.sum(axis=1).rolling(window=window, min_periods=1).mean()
            )

    return pd.DataFrame(feature_data, index=delays.index).astype("float32")


def attach_network_features(
    flights: pd.DataFrame, hourly_features: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Join each flight only to features from its previous completed UTC hour."""
    result = flights.copy()
    result["network_anchor_utc"] = (
        result["scheduled_utc"].dt.floor("h") - pd.Timedelta(hours=1)
    )
    feature_names = hourly_features.columns.tolist()
    hourly = hourly_features.rename_axis("network_anchor_utc").reset_index()
    result = result.merge(hourly, on="network_anchor_utc", how="left", validate="many_to_one")
    result["network_feature_coverage"] = result[feature_names].notna().any(axis=1)
    return result, feature_names


def build_network_dataset(
    config_path: str | Path = "config/default.yaml",
) -> dict[str, Any]:
    """Create and persist the approved flight-level network feature table."""
    config = load_config(config_path)
    network = config["network_features"]
    delays, activity = load_hourly_sources(config)
    hourly = build_hourly_network_features(
        delays,
        activity,
        european_airports=network["european_airports"],
        balkan_airports=network["balkan_airports"],
        lags_hours=network["lags_hours"],
        rolling_windows_hours=network["rolling_windows_hours"],
    )
    base_path = Path(config["data"]["processed_dir"]) / "balkan_flights.parquet"
    flights = pd.read_parquet(base_path)
    enriched, feature_names = attach_network_features(flights, hourly)
    output = Path(config["data"]["network_dataset"])
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)

    return {
        "records": int(len(enriched)),
        "network_features": len(feature_names),
        "covered_records": int(enriched["network_feature_coverage"].sum()),
        "uncovered_records": int((~enriched["network_feature_coverage"]).sum()),
        "earliest_source_hour": delays.index.min().isoformat(),
        "latest_source_hour": delays.index.max().isoformat(),
        "output": str(output),
    }
