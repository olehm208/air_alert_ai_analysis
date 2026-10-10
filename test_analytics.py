"""Focused boundary/coverage tests for dashboard analytics."""
from datetime import date

import pandas as pd
import pytest

from analytics import (
    aggregate_counts, available_date_range, confirmed_durations,
    count_by_region, daily_counts, filter_alerts, hourly_weekday_heatmap,
    kyiv_date_bounds, source_level_monthly,
)
from data_loader import REGIONS


def tiny_history():
    return pd.DataFrame({
        "region": ["Kyiv City", "Kyiv City", "Luhanska oblast"],
        "level": ["oblast", "oblast", "oblast"],
        "started_at": pd.to_datetime([
            "2026-03-28T22:30:00Z",  # Kyiv Mar 29, 00:30, EET
            "2026-03-29T21:30:00Z",  # Kyiv Mar 30, 00:30, EEST
            "2026-03-30T10:30:00Z",
        ], utc=True),
        "duration_minutes": [12.0, 30.0, 40.0],
        "estimated_end": [False, True, False],
        "end_unrecorded": [False, False, False],
    })


def test_daylight_saving_filter_uses_kyiv_calendar_days():
    data = tiny_history()
    mar29 = filter_alerts(data, date(2026, 3, 29), date(2026, 3, 29))
    assert len(mar29) == 1
    assert mar29.iloc[0]["started_at"] == pd.Timestamp("2026-03-28T22:30Z")
    lo, hi = kyiv_date_bounds(date(2026, 3, 29), date(2026, 3, 29))
    assert (hi - lo) == pd.Timedelta(hours=23)


def test_zero_records_date_is_not_dropped():
    result = daily_counts(tiny_history(), date(2026, 3, 29), date(2026, 3, 31))
    assert result["records"].tolist() == [1, 2, 0]
    assert aggregate_counts(result, "Daily")["records"].sum() == 3
    assert aggregate_counts(result, "Weekly")["records"].sum() == 3
    assert aggregate_counts(result, "Monthly")["records"].sum() == 3


def test_estimated_duration_excluded_and_unknown_regions_preserved():
    data = tiny_history()
    assert confirmed_durations(data).tolist() == [12.0, 40.0]
    regions = count_by_region(data)
    assert "Crimea (Autonomous Republic)" in regions["region"].tolist()
    assert regions.set_index("region").loc["Crimea (Autonomous Republic)", "records"] == 0
    assert len(REGIONS) == 27


def test_hourly_and_reporting_mix():
    data = tiny_history()
    h = hourly_weekday_heatmap(data)
    assert h.shape == (7, 24)
    assert h.to_numpy().sum() == len(data)
    assert source_level_monthly(data)["records"].sum() == len(data)


def test_invalid_date_order():
    with pytest.raises(ValueError):
        kyiv_date_bounds(date(2026, 4, 2), date(2026, 4, 1))
