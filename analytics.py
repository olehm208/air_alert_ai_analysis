"""Reproducible retrospective statistics for AirAlert's Analytics tab.

Alert rows are *records*, not distinct military events. In particular, official
oblast/raion/hromada records may overlap. Missing rows are not a 'safe' signal.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data_loader import REGIONS

KYIV_TZ = "Europe/Kyiv"


def kyiv_date_bounds(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return [start, day after end) as UTC instants, DST-safe."""
    if end < start:
        raise ValueError("end date must not precede start date")
    lo = pd.Timestamp(start).tz_localize(KYIV_TZ).tz_convert("UTC")
    hi = pd.Timestamp(end + timedelta(days=1)).tz_localize(KYIV_TZ).tz_convert("UTC")
    return lo, hi


def filter_alerts(
    alerts: pd.DataFrame,
    start: date,
    end: date,
    region: str | None = None,
    level: str | None = None,
) -> pd.DataFrame:
    """Select rows whose *start* occurred during the Kyiv local date range.

    This is a retrospective start-date filter, not a point-in-time feature.
    It intentionally does not claim whether an alert was active on any day.
    """
    lo, hi = kyiv_date_bounds(start, end)
    mask = alerts["started_at"].ge(lo) & alerts["started_at"].lt(hi)
    if region is not None:
        mask &= alerts["region"].eq(region).fillna(False)
    if level is not None:
        mask &= alerts["level"].eq(level).fillna(False)
    return alerts.loc[mask].copy()


def available_date_range(alerts: pd.DataFrame) -> tuple[date, date]:
    if alerts.empty:
        raise ValueError("dataset has no alert rows")
    local = alerts["started_at"].dt.tz_convert(KYIV_TZ)
    return local.min().date(), local.max().date()


def daily_counts(filtered: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """All local calendar dates, including dates with no *recorded starts*."""
    dates = pd.date_range(start=start, end=end, freq="D")
    if filtered.empty:
        counts = pd.Series(dtype="int64")
    else:
        local_dates = filtered["started_at"].dt.tz_convert(KYIV_TZ).dt.date
        counts = local_dates.value_counts()
    values = pd.Series(dates.date, name="date").map(counts).fillna(0).astype(int)
    return pd.DataFrame({"date": dates, "records": values.to_numpy()})


def aggregate_counts(daily: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Daily, calendar weekly (Monday start), or calendar monthly counts."""
    if cadence == "Daily":
        return daily.copy()
    if cadence == "Weekly":
        periods = daily["date"].dt.to_period("W-SUN")
    elif cadence == "Monthly":
        periods = daily["date"].dt.to_period("M")
    else:
        raise ValueError(f"unsupported cadence: {cadence}")
    result = daily.groupby(periods, sort=True)["records"].sum()
    return pd.DataFrame({"date": result.index.start_time, "records": result.to_numpy()})


def hourly_weekday_heatmap(filtered: pd.DataFrame) -> pd.DataFrame:
    """7-by-24 matrix of recorded starts in Kyiv time (Mon=0)."""
    if filtered.empty:
        return pd.DataFrame(np.zeros((7, 24), dtype=int), index=range(7), columns=range(24))
    local = filtered["started_at"].dt.tz_convert(KYIV_TZ)
    counts = pd.crosstab(local.dt.dayofweek, local.dt.hour)
    return counts.reindex(index=range(7), columns=range(24), fill_value=0)


def count_by_region(filtered: pd.DataFrame) -> pd.DataFrame:
    """Retain regions without rows instead of silently dropping them."""
    counts = filtered["region"].value_counts(dropna=True)
    regions = sorted(set(REGIONS) | set(counts.index))
    result = pd.Series(regions, name="region").to_frame()
    result["records"] = result["region"].map(counts).fillna(0).astype(int)
    return result.sort_values(["records", "region"], ascending=[False, True]).reset_index(drop=True)


def confirmed_durations(filtered: pd.DataFrame) -> pd.Series:
    """Exclude estimated volunteer endings and missing official endings."""
    good = (
        (~filtered["estimated_end"])
        & (~filtered["end_unrecorded"])
        & filtered["duration_minutes"].notna()
        & filtered["duration_minutes"].ge(0)
    )
    return filtered.loc[good, "duration_minutes"].astype(float)


def source_level_monthly(filtered: pd.DataFrame) -> pd.DataFrame:
    """Official reporting-level mix per Kyiv calendar month."""
    if filtered.empty:
        return pd.DataFrame(columns=["month", "level", "records"])
    local = filtered["started_at"].dt.tz_convert(KYIV_TZ)
    df = pd.DataFrame({
        "month": local.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp(),
        "level": filtered["level"].fillna("unknown"),
    })
    return df.groupby(["month", "level"], as_index=False).size().rename(columns={"size": "records"})
