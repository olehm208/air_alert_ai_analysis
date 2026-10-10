"""Leakage-aware retrospective forecasts of *recorded* air-alert starts.

Forecast time t is the beginning of a UTC hour. Input alert starts are in
[t-... , t) only. Label y[t] concerns [t,t+h). A label can enter a model or
rolling baseline only at/after t+h. Walk-forward folds are disjoint 28-day
windows; each fold chooses ML architecture on an earlier validation period.

The CSVs lack message publication/ingestion timestamps: the code prevents
mathematical look-ahead, but cannot prove that historical Telegram records
were available to users at exactly their event timestamps.
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
N_WALK_FORWARD_FOLDS = 4
MIN_RECENT_STARTS = 40
LAGS = (1, 3, 6, 12, 24, 72, 168, 672)
FEATURES = [
    *(f"starts_prior_{n}h" for n in LAGS),
    "hours_since_start", "hour_sin", "hour_cos", "weekday_sin",
    "weekday_cos", "year_sin", "year_cos",
]

BASELINES = {
    "past_overall": "Past overall frequency",
    "hour_of_day": "Past hour-of-day frequency",
    "rolling_7d": "Trailing 7-day frequency",
    "rolling_30d": "Trailing 30-day frequency",
}


class InsufficientDataError(ValueError):
    """Insufficient usable observations to produce a defensible replay."""


@dataclass
class ForecastReport:
    source: str
    region: str
    horizon: int
    origin: pd.Timestamp
    latest_record: pd.Timestamp
    probability: float
    model_name: str
    metrics: dict[str, float | None]  # latest untouched test
    baseline_metrics: dict[str, float | None]  # compatibility: hour-of-day
    all_baseline_metrics: dict[str, dict[str, float | None]]
    validation_scores: dict[str, float]
    test: pd.DataFrame  # latest fold, non-overlapping windows
    walk_forward: pd.DataFrame  # four chronological, disjoint folds
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
    """Create regional hourly start counts, ending at a complete UTC hour."""
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
    """Build strictly-past input features and labels with precise future windows."""
    _check_horizon(horizon)
    if starts.empty or starts.index.tz is None:
        raise ValueError("starts needs a nonempty UTC datetime index")
    if not starts.index.is_monotonic_increasing:
        raise ValueError("starts must be sorted")
    idx = starts.index.append(pd.DatetimeIndex([starts.index[-1] + pd.Timedelta(hours=1)]))
    available = starts.reindex(idx, fill_value=0)
    prior = available.shift(1, fill_value=0)
    data = pd.DataFrame(index=idx)
    for n in LAGS:
        data[f"starts_prior_{n}h"] = prior.rolling(n, min_periods=1).sum().to_numpy()
    positions = np.arange(len(idx))
    most_recent_prior = np.maximum.accumulate(
        np.where(prior.to_numpy() > 0, positions - 1, -100000)
    )
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
    """No label is used until its full forecast horizon has elapsed."""
    _check_horizon(horizon)
    origin = pd.Timestamp(origin)
    test_start = origin - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=validation_days)
    duration = pd.Timedelta(hours=horizon)
    beginnings = examples.index
    after_warmup = beginnings >= beginnings.min() + pd.Timedelta(hours=WARMUP_HOURS)
    known_labels = examples["target"].notna()
    train = examples.loc[after_warmup & known_labels & ((beginnings + duration) <= val_start)]
    val = examples.loc[known_labels & (beginnings >= val_start) & ((beginnings + duration) <= test_start)]
    test = examples.loc[known_labels & (beginnings >= test_start) & ((beginnings + duration) <= origin)]
    # Non-overlapping validation/test targets, for interpretable evaluation.
    val = val.iloc[::horizon]
    test = test.iloc[::horizon]
    boundaries = {
        "train_labels_end": val_start,
        "validation_labels_end": test_start,
        "test_labels_end": origin,
        "train_origins_end": val_start - duration,
    }
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
    """Only pre-test labels determine reference rates."""
    overall = float(train["target"].mean())
    g = train.groupby("local_hour")["target"].agg(["sum", "count"])
    smoothed = (g["sum"] + 24 * overall) / (g["count"] + 24)
    return overall, smoothed.to_dict()


def _baseline_predict(test: pd.DataFrame, parameters: tuple[float, dict[int, float]]) -> np.ndarray:
    overall, hours = parameters
    return test["local_hour"].map(hours).fillna(overall).to_numpy(dtype=float)


def rolling_prior_probabilities(
    examples: pd.DataFrame, horizon: int, prior_rate: float, days: int,
    smoothing: int = 24,
) -> pd.Series:
    """Rolling *completed* label frequency available at every forecast origin.

    A label at s requires observations in [s, s+h). It is therefore only
    allowed into a forecast at t when s+h <= t. Shift by h hours *before*
    rolling. This includes legitimately matured earlier holdout labels and
    supports on-the-fly updating without knowing future outcomes.
    """
    _check_horizon(horizon)
    if not 0 <= prior_rate <= 1 or days < 1:
        raise ValueError("Invalid baseline parameters")
    matured = examples["target"].shift(horizon)
    count = matured.rolling(days * 24, min_periods=1).count()
    positives = matured.rolling(days * 24, min_periods=1).sum()
    result = (positives.fillna(0) + smoothing * prior_rate) / (count + smoothing)
    result.name = f"rolling_{days}d"
    return result


def _fold_evaluation(
    examples: pd.DataFrame, end: pd.Timestamp, horizon: int,
) -> dict[str, Any]:
    """Independently validate, refit and score one disjoint test fold."""
    train, val, test, bounds = temporal_splits(examples, end, horizon)
    if len(train) < 500 or len(val) < 25 or len(test) < 50:
        raise InsufficientDataError("Not enough train/validation/test windows")
    if train["target"].nunique() < 2 or (train["target"] == 1).sum() < 15:
        raise InsufficientDataError("Too few distinct positive training outcomes")

    candidates = _candidates()
    val_scores = {}
    for name, clf in candidates.items():
        clf.fit(train[FEATURES], train["target"].astype(int))
        val_p = clf.predict_proba(val[FEATURES])[:, 1]
        val_scores[name] = float(brier_score_loss(val["target"].astype(int), val_p))
    selected = min(val_scores, key=val_scores.get)
    # All rows in refit are fully matured by the first test origin.
    refit = examples.loc[
        (examples.index >= examples.index.min() + pd.Timedelta(hours=WARMUP_HOURS))
        & examples["target"].notna()
        & ((examples.index + pd.Timedelta(hours=horizon)) <= bounds["validation_labels_end"])
    ]
    if refit["target"].nunique() < 2:
        raise InsufficientDataError("Only one target class in train+validation")
    model = clone(candidates[selected])
    model.fit(refit[FEATURES], refit["target"].astype(int))
    pred = model.predict_proba(test[FEATURES])[:, 1]
    overall, hours = _baseline_fit(refit)
    all_predictions = {
        "past_overall": np.full(len(test), overall),
        "hour_of_day": _baseline_predict(test, (overall, hours)),
        "rolling_7d": rolling_prior_probabilities(examples, horizon, overall, 7).loc[test.index].to_numpy(),
        "rolling_30d": rolling_prior_probabilities(examples, horizon, overall, 30).loc[test.index].to_numpy(),
    }
    y = test["target"].astype(int)
    scores = _score(y, pred)
    baselines = {key: _score(y, values) for key, values in all_predictions.items()}
    comparison = pd.DataFrame({
        "origin_utc": test.index,
        "actual": y.to_numpy(),
        "model_probability": pred,
        **{f"{key}_probability": val for key, val in all_predictions.items()},
    })
    # Legacy column used by v2 exports.
    comparison["baseline_probability"] = comparison["hour_of_day_probability"]
    comparison["origin_kyiv"] = comparison["origin_utc"].dt.tz_convert("Europe/Kyiv")
    summary = {
        "fold_start": bounds["validation_labels_end"],
        "fold_end": end,
        "test_samples": len(test),
        "selected_model": selected,
        "positive_rate": scores["positive_rate"],
        "brier_model": scores["brier"],
        **{f"brier_{key}": s["brier"] for key, s in baselines.items()},
    }
    return {
        "summary": summary, "scores": scores, "baselines": baselines,
        "comparison": comparison, "validation_scores": val_scores,
        "selected": selected, "train": train, "val": val, "test": test,
        "boundaries": bounds,
    }


def train_and_evaluate(alerts: pd.DataFrame, region: str, horizon: int) -> ForecastReport:
    """Walk-forward evaluation on four consecutive folds, latest = primary.

    Selection is validation-only per fold. Test data is not used to choose the
    classifier. Latest snapshot refit is separate from all held-out scores.
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

    fold_results = []
    for i in range(N_WALK_FORWARD_FOLDS - 1, -1, -1):
        fold_end = origin - pd.Timedelta(days=TEST_DAYS * i)
        fold_results.append(_fold_evaluation(examples, fold_end, horizon))
    latest = fold_results[-1]
    selected = latest["selected"]

    # This replay model MAY use the latest matured test labels: it is trained
    # only *after* all test evaluation, and never supplies holdout predictions.
    known = examples.loc[
        (examples.index >= examples.index.min() + pd.Timedelta(hours=WARMUP_HOURS))
        & examples["target"].notna()
        & ((examples.index + pd.Timedelta(hours=horizon)) <= origin)
    ]
    model = clone(_candidates()[selected])
    model.fit(known[FEATURES], known["target"].astype(int))
    replay_probability = float(model.predict_proba(examples.loc[[origin], FEATURES])[0, 1])

    return ForecastReport(
        source=str(alerts["source"].iloc[0]), region=region, horizon=horizon,
        origin=origin, latest_record=last_record, probability=replay_probability,
        model_name=selected, metrics=latest["scores"],
        baseline_metrics=latest["baselines"]["hour_of_day"],
        all_baseline_metrics=latest["baselines"],
        validation_scores=latest["validation_scores"],
        test=latest["comparison"],
        walk_forward=pd.DataFrame([f["summary"] for f in fold_results]),
        periods=latest["boundaries"], recent_starts=total_recent,
        train_samples=len(latest["train"]),
        validation_samples=len(latest["val"]),
        test_samples=len(latest["test"]),
    )
