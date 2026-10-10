"""Regression tests for parsing, duplicate removal, and unavailable coverage."""
from pathlib import Path

import pandas as pd

from data_loader import load_alerts, region_coverage


def test_new_crimea_records_are_supported(tmp_path: Path) -> None:
    pd.DataFrame([
        {
            "region": "Autonomous Republic of Crimea",
            "started_at": "2026-10-11T10:00:00+00:00",
            "finished_at": "2026-10-11T11:00:00+00:00",
            "naive": False,
        },
        {
            "region": "Sevastopol",
            "started_at": "2026-10-11T12:00:00+00:00",
            "finished_at": None,
            "naive": False,
        },
    ]).to_csv(tmp_path / "volunteer_data_en.csv", index=False)

    events = load_alerts("volunteer", tmp_path)
    cov = region_coverage(events).set_index("region")
    assert cov.loc["Crimea (Autonomous Republic)", "records"] == 1
    assert cov.loc["Sevastopol City", "records"] == 1
    assert cov.loc["Luhanska oblast", "coverage"] == "No records"
    assert bool(events.loc[events["region"] == "Sevastopol City", "end_unrecorded"].iloc[0])
    assert events.loc[events["region"] == "Crimea (Autonomous Republic)", "duration_minutes"].iloc[0] == 60


def test_exact_official_duplicates_removed(tmp_path: Path) -> None:
    row = {
        "oblast": "Luhanska oblast",
        "raion": None,
        "hromada": None,
        "level": "oblast",
        "started_at": "2022-03-24T11:28:27+00:00",
        "finished_at": "2022-03-28T03:22:00+00:00",
        "source": "official",
    }
    pd.DataFrame([row, row]).to_csv(tmp_path / "official_data_en.csv", index=False)
    events = load_alerts("official", tmp_path)
    assert len(events) == 1
