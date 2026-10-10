"""Checks for new prequential baselines and walk-forward leakage boundaries."""
import numpy as np
import pandas as pd
import pytest

from forecasting import (
    BASELINES, _fold_evaluation, build_examples, rolling_prior_probabilities,
    temporal_splits, TEST_DAYS, VALIDATION_DAYS, FEATURES,
)


def example_starts():
    idx = pd.date_range("2026-01-01", periods=24 * 240, freq="h", tz="UTC")
    rng = np.random.default_rng(723)
    # Pattern changes through time, but remains non-degenerate across folds.
    p = np.where(np.arange(len(idx)) < 24 * 140, 0.12, 0.22)
    return pd.Series((rng.random(len(idx)) < p).astype(int), index=idx)


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_rolling_baseline_cannot_see_future_event(horizon):
    starts = example_starts()
    t = starts.index[24 * 120]
    first = build_examples(starts, horizon)
    first_probs = rolling_prior_probabilities(first, horizon, 0.35, 7)
    changed = starts.copy()
    # Adds a start AFTER forecast origin, potentially within target horizon.
    future_idx = starts.index.get_loc(t) + 1
    changed.iloc[future_idx] += 1
    second = build_examples(changed, horizon)
    second_probs = rolling_prior_probabilities(second, horizon, 0.35, 7)
    assert first_probs.loc[t] == pytest.approx(second_probs.loc[t])
    pd.testing.assert_series_equal(first.loc[t, FEATURES], second.loc[t, FEATURES])
    assert second.loc[t, "target"] >= first.loc[t, "target"]


def test_matured_history_is_allowed_to_update_rolling_baseline():
    idx = pd.date_range("2026-01-01", periods=24 * 45, freq="h", tz="UTC")
    starts = pd.Series(0, index=idx)
    t = idx[24 * 30]
    p0 = rolling_prior_probabilities(build_examples(starts, 6), 6, .2, 7).at[t]
    starts.loc[idx[24 * 30 - 1]] = 1  # 1h before t; past and fully observed at t
    p1 = rolling_prior_probabilities(build_examples(starts, 6), 6, .2, 7).at[t]
    assert p1 > p0
    assert p1 < 1.0


def test_four_walkforward_windows_do_not_overlap():
    starts = example_starts()
    horizon = 6
    examples = build_examples(starts, horizon)
    origin = starts.index[-1] + pd.Timedelta(hours=1)
    ends = [origin - pd.Timedelta(days=TEST_DAYS * i) for i in (3, 2, 1, 0)]
    windows = [temporal_splits(examples, x, horizon) for x in ends]
    for i, (train, val, test, bounds) in enumerate(windows):
        assert len(train) >= 500
        assert len(val) >= 25
        assert len(test) >= 50
        assert (train.index + pd.Timedelta(hours=horizon) <= bounds["train_labels_end"]).all()
        assert (val.index + pd.Timedelta(hours=horizon) <= bounds["validation_labels_end"]).all()
        assert (test.index + pd.Timedelta(hours=horizon) <= bounds["test_labels_end"]).all()
        if i:
            previous = windows[i - 1][2]
            assert previous.index.max() + pd.Timedelta(hours=horizon) <= test.index.min()
        assert bounds["validation_labels_end"] == ends[i] - pd.Timedelta(days=TEST_DAYS)
        assert bounds["train_labels_end"] == ends[i] - pd.Timedelta(days=TEST_DAYS + VALIDATION_DAYS)


def test_earlier_fold_immune_to_later_changes():
    starts = example_starts()
    end = starts.index[-1] + pd.Timedelta(hours=1) - pd.Timedelta(days=56)
    h = 6
    original = _fold_evaluation(build_examples(starts, h), end, h)
    changed = starts.copy()
    changed.loc[end + pd.Timedelta(hours=10)] += 1
    result = _fold_evaluation(build_examples(changed, h), end, h)
    assert result["selected"] == original["selected"]
    assert result["validation_scores"] == original["validation_scores"]
    pd.testing.assert_frame_equal(result["comparison"], original["comparison"])
    assert set(original["baselines"]) == set(BASELINES)


def test_rolling_baseline_missing_history_uses_pretest_prior():
    idx = pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC")
    ex = build_examples(pd.Series(0, index=idx), 6)
    p = rolling_prior_probabilities(ex, 6, .4, 30)
    assert p.iloc[0] == pytest.approx(.4)
    assert 0 <= p.iloc[-1] <= 1
