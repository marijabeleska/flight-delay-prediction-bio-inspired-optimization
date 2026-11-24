import numpy as np
import pandas as pd

from flight_delay_bioopt.network_features import (
    attach_network_features,
    build_hourly_network_features,
)


def test_network_features_use_only_previous_completed_hours() -> None:
    index = pd.date_range("2023-01-01T00:00:00Z", periods=4, freq="h")
    delays = pd.DataFrame({"EDDF": [10, 20, 30, 40], "LWSK": [1, 2, 3, 4]}, index=index)
    activity = pd.DataFrame({"EDDF": [1, 1, 1, 1], "LWSK": [1, 1, 1, 1]}, index=index)
    hourly = build_hourly_network_features(
        delays,
        activity,
        european_airports=["EDDF"],
        balkan_airports=["LWSK"],
        lags_hours=[1, 2, 3],
        rolling_windows_hours=[3],
    )
    flights = pd.DataFrame(
        {"scheduled_utc": pd.to_datetime(["2023-01-01T03:30:00Z"], utc=True)}
    )
    enriched, _ = attach_network_features(flights, hourly)

    assert enriched.loc[0, "network_delay_EDDF_lag_1h"] == 30
    assert enriched.loc[0, "network_delay_EDDF_lag_2h"] == 20
    assert enriched.loc[0, "network_delay_EDDF_lag_3h"] == 10
    assert np.isclose(enriched.loc[0, "network_europe_mean_delay_rolling_3h"], 20)


def test_inactive_hour_is_not_treated_as_zero_delay() -> None:
    index = pd.date_range("2023-01-01T00:00:00Z", periods=2, freq="h")
    delays = pd.DataFrame({"EDDF": [0, 20], "LWSK": [0, 2]}, index=index)
    activity = pd.DataFrame({"EDDF": [0, 1], "LWSK": [0, 1]}, index=index)
    hourly = build_hourly_network_features(
        delays,
        activity,
        european_airports=["EDDF"],
        balkan_airports=["LWSK"],
        lags_hours=[1],
        rolling_windows_hours=[2],
    )

    assert np.isnan(hourly.iloc[0]["network_delay_EDDF_lag_1h"])
    assert hourly.iloc[0]["network_active_EDDF_lag_1h"] == 0
