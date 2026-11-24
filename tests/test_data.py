from pathlib import Path

import numpy as np

from flight_delay_bioopt.data import load_airport_file


def test_load_airport_file_builds_target_and_time_features(tmp_path: Path) -> None:
    rows = np.array(
        [
            ["LSZH", "LWSK", "1578268800", "0"],
            ["EDDF", "LWSK", "1578272400", "1200"],
        ],
        dtype="U32",
    )
    path = tmp_path / "Delays_LWSK.npy"
    np.save(path, rows)

    frame = load_airport_file(path)

    assert frame["is_delayed"].tolist() == [0, 1]
    assert frame["route"].tolist() == ["LSZH_LWSK", "EDDF_LWSK"]
    assert frame["year"].tolist() == [2020, 2020]
    assert frame["destination_file_mismatch"].sum() == 0
    assert frame["has_missing_core"].sum() == 0


def test_load_airport_file_flags_duplicates_and_extremes(tmp_path: Path) -> None:
    rows = np.array(
        [
            ["EDDF", "LWSK", "1578268800", "50000"],
            ["EDDF", "LWSK", "1578268800", "50000"],
        ],
        dtype="U32",
    )
    path = tmp_path / "Delays_LWSK.npy"
    np.save(path, rows)

    frame = load_airport_file(path)

    assert frame["is_exact_duplicate"].sum() == 2
    assert frame["is_extreme_delay"].sum() == 2
