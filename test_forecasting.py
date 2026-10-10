"""Checks that forecasts cannot look ahead of their UTC forecast origin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import (
    FEATURES, InsufficientDataError, build_examples, build_hourly_events,
    temporal_splits, train_and_evaluate,
)


def mini_alerts() -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["volunteer"] * 5,
        "region": ["Kyiv City"] * 5,
        "started_at": pd.to_datetime([
            "2026-09-01 00:17Z", "2026-09-01 01:34Z", "2026-09-01 05:01Z",
            "2026-09-01 06:10Z", "2026-09-01 10:20Z"
        ], utc=True),
        "finished_at": pd.to_datetime([
            "2026-09-01 00:40Z", "2026-09-01 02:10Z", "2026-09-01 05:34Z",
            "2026-09-01 07:10Z", "2026-09-01 10:55Z",
        ], utc=True),
    })


def test_features_only_use_strictly_past_start_hours():
    alerts = mini_alerts()
    hourly = build_hourly_events(alerts, "Kyiv City", pd.Timestamp("2026-09-01 10:00Z"))
    examples = build_examples(hourly, 3)
    t = pd.Timestamp("2026-09-01 05:00Z")
    # 00:17 and 01:34 are known at 05:00; the 05:01 alert is NOT.
    assert examples.at[t, "starts_prior_6h"] == 2
    assert examples.at[t, "starts_prior_1h"] == 0
    assert examples.at[t, "target"] == 1  # event within [05:00,08:00)


def test_future_changes_do_not_modify_past_features_or_completed_labels():
    alert = mini_alerts()
    origin = pd.Timestamp("2026-09-01 10:00Z")
    a = build_examples(build_hourly_events(alert, "Kyiv City", origin), 3)
    # Add an extra 08:00 event; 05:00 features are unchanged and target ends 08:00.
    row = alert.iloc[[0]].copy()
    row["started_at"] = pd.Timestamp("2026-09-01 08:25Z")
    changed = pd.concat([alert, row], ignore_index=True)
    b = build_examples(build_hourly_events(changed, "Kyiv City", origin), 3)
    t = pd.Timestamp("2026-09-01 05:00Z")
    pd.testing.assert_series_equal(a.loc[t, FEATURES], b.loc[t, FEATURES])
    assert a.loc[t, "target"] == b.loc[t, "target"]
    # A future event at 08:00 can change the 07:00 target but not its features.
    t2 = pd.Timestamp("2026-09-01 07:00Z")
    pd.testing.assert_series_equal(a.loc[t2, FEATURES], b.loc[t2, FEATURES])
    assert a.loc[t2, "target"] == 0
    assert b.loc[t2, "target"] == 1


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_exact_future_label_boundaries(horizon):
    n = 2000
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    a = pd.Series(np.zeros(n, dtype=int), index=idx)
    at = 1000
    a.iloc[at + horizon - 1] = 1
    a.iloc[at - 1] = 1
    ex = build_examples(a, horizon)
    assert ex.at[idx[at], "target"] == 1
    assert ex.at[idx[at], "starts_prior_1h"] == 1
    assert ex.at[idx[-1], "target"] != ex.at[idx[-1], "target"] if horizon > 1 else True
    assert pd.isna(ex.at[idx[-1] + pd.Timedelta(hours=1), "target"])


def test_train_validation_test_labels_fully_mature_before_cutoffs():
    n = 24 * 120
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    starts = pd.Series((np.arange(n) % 8 == 0).astype(int), index=idx)
    horizon = 6
    examples = build_examples(starts, horizon)
    observation_end = idx[-1] + pd.Timedelta(hours=1)
    train, val, test, bounds = temporal_splits(examples, observation_end, horizon)
    assert len(train) > 100
    assert not val.empty and not test.empty
    assert all(train.index + pd.Timedelta(hours=horizon) <= bounds["train_labels_end"])
    assert all(val.index >= bounds["train_labels_end"])
    assert all(val.index + pd.Timedelta(hours=horizon) <= bounds["validation_labels_end"])
    assert all(test.index >= bounds["validation_labels_end"])
    assert all(test.index + pd.Timedelta(hours=horizon) <= observation_end)
    assert all((test.index[1:] - test.index[:-1]) == pd.Timedelta(hours=horizon))


def test_missing_region_has_no_fabricated_probability():
    with pytest.raises(InsufficientDataError):
        train_and_evaluate(mini_alerts(), "Crimea (Autonomous Republic)", 6)


def test_invalid_horizon():
    idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        build_examples(pd.Series(0, index=idx), 4)
