"""Shared, source-aware loader for the air-alert dashboard.

This is retrospective interval data, NOT a real-time status feed. An absent row
cannot be interpreted as no alert, especially for regions with missing coverage.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "dataset_repo" / "ukrainian-air-raid-sirens-dataset" / "datasets"
)

FILES = {
    "volunteer": "volunteer_data_en.csv",
    "official": "official_data_en.csv",
}

# The 24 oblasts, Kyiv City, the Autonomous Republic of Crimea, and Sevastopol.
REGIONS = (
    "Cherkaska oblast",
    "Chernihivska oblast",
    "Chernivetska oblast",
    "Crimea (Autonomous Republic)",
    "Dnipropetrovska oblast",
    "Donetska oblast",
    "Ivano-Frankivska oblast",
    "Kharkivska oblast",
    "Khersonska oblast",
    "Khmelnytska oblast",
    "Kirovohradska oblast",
    "Kyiv City",
    "Kyivska oblast",
    "Luhanska oblast",
    "Lvivska oblast",
    "Mykolaivska oblast",
    "Odeska oblast",
    "Poltavska oblast",
    "Rivnenska oblast",
    "Sevastopol City",
    "Sumska oblast",
    "Ternopilska oblast",
    "Vinnytska oblast",
    "Volynska oblast",
    "Zakarpatska oblast",
    "Zaporizka oblast",
    "Zhytomyrska oblast",
)

ALIASES = {
    "Crimea": "Crimea (Autonomous Republic)",
    "Autonomous Republic of Crimea": "Crimea (Autonomous Republic)",
    "Avtonomna Respublika Krym": "Crimea (Autonomous Republic)",
    "Sevastopol": "Sevastopol City",
    "Sevastopol city": "Sevastopol City",
    "Luhansk oblast": "Luhanska oblast",
}

COLUMNS = [
    "source", "region", "level", "raion", "hromada", "started_at",
    "finished_at", "duration_minutes", "estimated_end", "end_unrecorded",
]


def load_alerts(source: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read and normalize one historical dataset; keep its provenance separate.

    Timestamp columns are timezone-aware UTC. A missing finish is retained as NaT;
    an estimated volunteer finish is retained but marked as estimated_end=True.
    """
    if source not in FILES:
        raise ValueError(f"source must be one of: {', '.join(FILES)}")

    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = folder / FILES[source]
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    raw = pd.read_csv(path)
    required = (
        {"region", "started_at", "finished_at", "naive"}
        if source == "volunteer"
        else {"oblast", "raion", "hromada", "level", "started_at", "finished_at"}
    )
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"{path.name}: missing columns: {sorted(missing_columns)}")

    # Official export contains exact duplicates, but potentially overlapping
    # district-level observations are NOT duplicates and must remain.
    raw = raw.drop_duplicates().copy()

    df = pd.DataFrame(index=raw.index)
    df["source"] = source
    df["region"] = raw["region"] if source == "volunteer" else raw["oblast"]
    df["region"] = df["region"].astype("string").str.strip().replace(ALIASES)
    df["level"] = "oblast" if source == "volunteer" else raw["level"]
    df["raion"] = pd.NA if source == "volunteer" else raw["raion"]
    df["hromada"] = pd.NA if source == "volunteer" else raw["hromada"]

    df["started_at"] = pd.to_datetime(raw["started_at"], utc=True, errors="coerce")
    df["finished_at"] = pd.to_datetime(raw["finished_at"], utc=True, errors="coerce")
    if df["started_at"].isna().any():
        raise ValueError(f"{path.name}: invalid or missing start timestamps")

    bad_end = raw["finished_at"].notna() & df["finished_at"].isna()
    if bad_end.any():
        raise ValueError(f"{path.name}: invalid end timestamps")
    if (df["finished_at"] < df["started_at"]).any():
        raise ValueError(f"{path.name}: some alerts end before they start")

    df["estimated_end"] = raw["naive"].astype(bool) if source == "volunteer" else False
    df["end_unrecorded"] = df["finished_at"].isna()
    df["duration_minutes"] = (
        (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
    )

    return df[COLUMNS].sort_values("started_at", kind="stable").reset_index(drop=True)


def region_coverage(alerts: pd.DataFrame) -> pd.DataFrame:
    """Include every region, even those that have no recorded observations.

    The status is about dataset coverage, NOT whether alerts are currently active.
    The 30-record cutoff merely marks sparse history; it is not ML validation.
    """
    if not set(COLUMNS).issubset(alerts.columns):
        raise ValueError("Expected normalized alerts from load_alerts()")

    stats = alerts.groupby("region", dropna=False).agg(
        records=("started_at", "size"),
        first_record=("started_at", "min"),
        latest_record=("started_at", "max"),
        estimated_finishes=("estimated_end", "sum"),
        unrecorded_finishes=("end_unrecorded", "sum"),
    )

    # Include unexpected future region spellings so that new data stays visible.
    all_regions = sorted(set(REGIONS) | set(alerts["region"].dropna().tolist()))
    stats = stats.reindex(all_regions)
    stats["records"] = stats["records"].fillna(0).astype(int)
    stats["estimated_finishes"] = stats["estimated_finishes"].fillna(0).astype(int)
    stats["unrecorded_finishes"] = stats["unrecorded_finishes"].fillna(0).astype(int)
    stats["coverage"] = "History available"
    stats.loc[stats["records"] < 30, "coverage"] = "Sparse history"
    stats.loc[stats["records"] == 0, "coverage"] = "No records"
    return stats.rename_axis("region").reset_index()


def kyiv_time(utc_timestamps: pd.Series) -> pd.Series:
    """Convert timestamps for display only; use UTC for model features."""
    return pd.to_datetime(utc_timestamps, utc=True).dt.tz_convert("Europe/Kyiv")
