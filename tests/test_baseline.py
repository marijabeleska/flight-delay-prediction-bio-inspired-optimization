import pandas as pd

from flight_delay_bioopt.baseline import (
    add_destination_local_features,
    build_pipeline,
    chronological_split,
    remove_repeated_copies,
)
from flight_delay_bioopt.data import load_config


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["EDDF", "EDDF", "LSZH", "EGLL"],
            "destination": ["LWSK"] * 4,
            "scheduled_utc": pd.to_datetime(
                [
                    "2021-12-31T23:30:00Z",
                    "2021-12-31T23:30:00Z",
                    "2022-06-01T12:00:00Z",
                    "2023-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "delay_minutes": [20.0, 20.0, 0.0, 30.0],
            "is_delayed": [1, 1, 0, 1],
        }
    )


def test_duplicate_removal_keeps_one_copy() -> None:
    cleaned, removed = remove_repeated_copies(_sample_frame())
    assert removed == 1
    assert len(cleaned) == 3


def test_local_features_and_chronological_boundaries() -> None:
    cleaned, _ = remove_repeated_copies(_sample_frame())
    featured = add_destination_local_features(cleaned, {"LWSK": "Europe/Skopje"})
    splits = chronological_split(featured, "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z")

    assert featured.iloc[0]["local_year"] == 2022
    assert featured.iloc[0]["local_hour"] == 0
    assert len(splits["train"]) == 1
    assert len(splits["validation"]) == 1
    assert len(splits["test"]) == 1


def test_balanced_pipeline_sets_class_weight() -> None:
    pipeline = build_pipeline(load_config(), class_weight="balanced")
    assert pipeline.named_steps["model"].class_weight == "balanced"
