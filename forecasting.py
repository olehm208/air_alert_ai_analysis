"""Source-specific, point-in-time alert-start forecasting.

Research-only forecasts about the *recording* of alert starts (not attacks or safety).
No alert finish times, future observations, or missing-region zeros are used.

A forecast origin is the beginning of a UTC hour. Input event counts use bins
strictly earlier than the origin; the target is any new recorded start within
[origin, origin + horizon). Fit/validation/test target windows end at or before
their own temporal boundaries. Test origins are sampled at horizon-hour stride
so overlapping outcome windows do not inflate the number of test samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 3, 6, 12)
LOOKBACK_DAYS = 240
WARMUP_HOURS = 28 * 24
VALIDATION_DAYS = 14
TEST_DAYS = 28
MIN_RECENT_STARTS = 40
LAGS = (1, 3, 6, 12, 24, 72, 168, 672)
FEATURES = [
    *(f"starts_prior_{n}h" for n in LAGS),
    "hours_since_start", "hour_sin", "hour_cos", "weekday_sin",
    "weekday_cos", "year_sin", "year_cos",
]


class InsufficientDataError(ValueError):
    """No defensible regional forecast can be estimated."""


@dataclass
class ForecastReport:
    source: str
    region: str
    horizon: int
    origin: pd.Timestamp
    latest_record: pd.Timestamp
    probability: float
    model_name: str
    metrics: dict[str, float | None]
    baseline_metrics: dict[str, float | None]
    validation_scores: dict[str, float]
    test: pd.DataFrame
    periods: dict[str, pd.Timestamp]
    recent_starts: int
    train_samples: int
    validation_samples: int
    test_samples: int


def _check_horizon(horizon: int) -> None:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")


def build_hourly_events(
    alerts: pd.DataFrame, region: str, observation_end: pd.Timestamp,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.Series:
    """Hourly *recorded starts*; no assertion that silent hours mean safety.

    Last interval is [observation_end-1h, observation_end); observation_end is
    an hour boundary determined from the latest start anywhere in the source.
    Incomplete source hours are not treated as observed.
    """
    observation_end = pd.Timestamp(observation_end)
    if observation_end.tzinfo is None:
        raise ValueError("observation_end must be timezone-aware")
    observation_end = observation_end.tz_convert("UTC")
    if observation_end != observation_end.floor("h"):
        raise ValueError("observation_end must be an exact UTC hour boundary")

    source_start = alerts["started_at"].min().floor("h")
    grid_start = max(source_start, observation_end - pd.Timedelta(days=lookback_days))
    if grid_start >= observation_end:
        raise InsufficientDataError("Not enough complete source history")
    idx = pd.date_range(grid_start, observation_end, freq="h", tz="UTC", inclusive="left")
    sub = alerts.loc[
        alerts["region"].eq(region).fillna(False)
        & alerts["started_at"].ge(grid_start)
        & alerts["started_at"].lt(observation_end),
        "started_at",
    ]
    starts = sub.dt.floor("h").value_counts().reindex(idx, fill_value=0).astype(int)
    starts.index.name = "origin"
    starts.name = "starts"
    return starts


def build_examples(starts: pd.Series, horizon: int) -> pd.DataFrame:
    """Construct features and labels with an explicit future cutoff.

    Input starts at t are not available until the next hour. The final row at
    index observation_end has features but has no label (future unknown).
    """
    _check_horizon(horizon)
    if starts.empty or starts.index.tz is None:
        raise ValueError("starts needs a nonempty UTC datetime index")
    if not starts.index.is_monotonic_increasing:
        raise ValueError("starts must be sorted")
    # Include an extra origin after the last fully observed hourly bin.
    idx = starts.index.append(pd.DatetimeIndex([starts.index[-1] + pd.Timedelta(hours=1)]))
    available = starts.reindex(idx, fill_value=0)
    prior = available.shift(1, fill_value=0)

    data = pd.DataFrame(index=idx)
    for n in LAGS:
        data[f"starts_prior_{n}h"] = prior.rolling(n, min_periods=1).sum().to_numpy()

    # Last event STRICTLY BEFORE t; never use the current hour as history.
    positions = np.arange(len(idx))
    most_recent_prior = np.maximum.accumulate(np.where(prior.to_numpy() > 0, positions - 1, -100000))
    data["hours_since_start"] = np.minimum(positions - 1 - most_recent_prior, 720)

    local = idx.tz_convert("Europe/Kyiv")
    hour = local.hour.to_numpy()
    weekday = local.dayofweek.to_numpy()
    yday = local.dayofyear.to_numpy()
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["year_sin"] = np.sin(2 * np.pi * yday / 365.25)
    data["year_cos"] = np.cos(2 * np.pi * yday / 365.25)
    data["local_hour"] = hour

    counts = available.to_numpy()
    # Targets [t,t+h), counting only bins with complete observed hours.
    # cumsum difference permits constant-time complete window aggregation.
    cs = np.r_[0, np.cumsum(counts)]
    k = np.arange(len(idx))
    end = k + horizon
    valid = end <= len(starts)
    data["target"] = np.nan
    data.loc[valid, "target"] = (cs[end[valid]] - cs[k[valid]] > 0).astype(int)
    return data


def temporal_splits(
    examples: pd.DataFrame, origin: pd.Timestamp, horizon: int,
    validation_days: int = VALIDATION_DAYS,
    test_days: int = TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Strict label maturation across train, validation and test boundaries."""
    _check_horizon(horizon)
    test_start = origin - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=validation_days)
    duration = pd.Timedelta(hours=horizon)
    beginnings = examples.index
    # Discard first 28 days of features to avoid partial initial rolling windows.
    after_warmup = beginnings >= beginnings.min() + pd.Timedelta(hours=WARMUP_HOURS)
    known_labels = examples["target"].notna()
    train = examples.loc[after_warmup & known_labels & ((beginnings + duration) <= val_start)]
    val = examples.loc[known_labels & (beginnings >= val_start) & ((beginnings + duration) <= test_start)]
    test = examples.loc[known_labels & (beginnings >= test_start) & ((beginnings + duration) <= origin)]
    # Score non-overlapping targets, even though training may include overlaps.
    val = val.iloc[::horizon]
    test = test.iloc[::horizon]
    boundaries = {"train_labels_end": val_start, "validation_labels_end": test_start,
                  "test_labels_end": origin, "train_origins_end": val_start - duration}
    return train, val, test, boundaries


def _candidates() -> dict[str, Any]:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=750, C=0.3, random_state=42)
        ),
        "Gradient-boosted trees": HistGradientBoostingClassifier(
            max_iter=110, max_leaf_nodes=13, learning_rate=0.045,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        ),
    }


def _score(y: pd.Series | np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    truth = np.asarray(y).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), 0.000001, 0.999999)
    has_both = len(np.unique(truth)) == 2
    return {
        "brier": float(brier_score_loss(truth, p)),
        "average_precision": float(average_precision_score(truth, p)) if has_both else None,
        "roc_auc": float(roc_auc_score(truth, p)) if has_both else None,
        "positive_rate": float(np.mean(truth)),
    }


def _baseline_fit(train: pd.DataFrame) -> tuple[float, dict[int, float]]:
    overall = float(train["target"].mean())
    g = train.groupby("local_hour")["target"].agg(["sum", "count"])
    smoothed = (g["sum"] + 24 * overall) / (g["count"] + 24)
    return overall, smoothed.to_dict()


def _baseline_predict(test: pd.DataFrame, parameters: tuple[float, dict[int, float]]) -> np.ndarray:
    overall, hours = parameters
    return test["local_hour"].map(hours).fillna(overall).to_numpy(dtype=float)


def train_and_evaluate(alerts: pd.DataFrame, region: str, horizon: int) -> ForecastReport:
    """Fit, select on validation, score untouched latest test, and refit for replay.

    The final replay origin is the last FULL source hour boundary, not 'now'.
    This cannot infer live alert status or safety from CSV data.
    """
    _check_horizon(horizon)
    if alerts.empty:
        raise InsufficientDataError("Empty source dataset")
    if region not in set(alerts["region"].dropna()):
        raise InsufficientDataError(f"{region}: no recorded history in this source")
    last_record = alerts["started_at"].max()
    origin = last_record.floor("h")
    starts = build_hourly_events(alerts, region, origin)
    total_recent = int(starts.sum())
    if total_recent < MIN_RECENT_STARTS:
        raise InsufficientDataError(
            f"Only {total_recent} regional starts in the most recent {LOOKBACK_DAYS} days "
            f"(minimum {MIN_RECENT_STARTS}). Coverage is too sparse for a forecast."
        )
    examples = build_examples(starts, horizon)
    train, val, test, boundaries = temporal_splits(examples, origin, horizon)
    if len(train) < 500 or len(val) < 25 or len(test) < 50:
        raise InsufficientDataError("Not enough independent training/validation/test samples")
    if train["target"].nunique() < 2 or (train["target"] == 1).sum() < 15:
        raise InsufficientDataError("Too few distinct training outcomes to fit a classifier")
    # Brier score remains defined for a one-class validation interval.
    # We show this instability rather than pretending the rare event is predictable.

    candidates = _candidates()
    val_scores = {}
    for name, clf in candidates.items():
        clf.fit(train[FEATURES], train["target"].astype(int))
        pred = clf.predict_proba(val[FEATURES])[:, 1]
        val_scores[name] = float(brier_score_loss(val["target"].astype(int), pred))
    selected = min(val_scores, key=val_scores.get)

    # The test period remains untouched throughout model selection.
    refit = examples.loc[
        (examples.index >= examples.index.min() + pd.Timedelta(hours=WARMUP_HOURS))
        & examples["target"].notna()
        & ((examples.index + pd.Timedelta(hours=horizon)) <= boundaries["validation_labels_end"])
    ]
    if refit["target"].nunique() < 2:
        raise InsufficientDataError("Only one target class in train+validation")
    test_model = clone(candidates[selected])
    test_model.fit(refit[FEATURES], refit["target"].astype(int))
    test_p = test_model.predict_proba(test[FEATURES])[:, 1]
    baseline_p = _baseline_predict(test, _baseline_fit(refit))
    scores = _score(test["target"], test_p)
    baseline_scores = _score(test["target"], baseline_p)

    # After *untouched test evaluation*, refit using only labels mature by
    # snapshot time. This model is NEVER the model used for reported test scores.
    full_known = examples.loc[
        (examples.index >= examples.index.min() + pd.Timedelta(hours=WARMUP_HOURS))
        & examples["target"].notna()
        & ((examples.index + pd.Timedelta(hours=horizon)) <= origin)
    ]
    replay_model = clone(candidates[selected])
    replay_model.fit(full_known[FEATURES], full_known["target"].astype(int))
    replay_features = examples.loc[[origin], FEATURES]
    replay_probability = float(replay_model.predict_proba(replay_features)[0, 1])

    comparison = pd.DataFrame({
        "origin_utc": test.index,
        "actual": test["target"].astype(int).to_numpy(),
        "model_probability": test_p,
        "baseline_probability": baseline_p,
    })
    comparison["origin_kyiv"] = comparison["origin_utc"].dt.tz_convert("Europe/Kyiv")
    return ForecastReport(
        source=str(alerts["source"].iloc[0]), region=region, horizon=horizon,
        origin=origin, latest_record=last_record, probability=replay_probability,
        model_name=selected, metrics=scores, baseline_metrics=baseline_scores,
        validation_scores=val_scores, test=comparison, periods=boundaries,
        recent_starts=total_recent, train_samples=len(train),
        validation_samples=len(val), test_samples=len(test),
    )
