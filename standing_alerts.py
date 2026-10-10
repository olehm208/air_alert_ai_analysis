"""Manually maintained long-running alert annotations, separate from historical CSV.

An annotation is NOT evidence of current threat status. It represents one
assumed continuous alert, not one new start in each selected time window.
It must never enter historical features, labels or test metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_loader import REGIONS

DEFAULT_PATH = Path(__file__).resolve().parent / "standing_alerts.json"
VALID_STATES = {"assumed_ongoing", "ended"}


def load_standing_context(path: str | Path | None = None) -> dict:
    """Load editable project annotations and reject invalid state strings."""
    p = Path(path) if path is not None else DEFAULT_PATH
    context = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(context, dict) or not isinstance(context.get("regions"), dict):
        raise ValueError("Expected an object with a regions mapping")
    if not isinstance(context.get("as_of_date"), str):
        raise ValueError("Standing alert annotations must have an as_of_date string")
    for region, state in context["regions"].items():
        if region not in REGIONS or state not in VALID_STATES:
            raise ValueError(f"Invalid standing alert annotation for {region!r}: {state!r}")
    return context


def ongoing_count(context: dict, region: str | None = None) -> int:
    """Count assumed ongoing statuses, independent of CSV, dates and source."""
    regions = context["regions"]
    if region is not None:
        return int(regions.get(region) == "assumed_ongoing")
    return sum(state == "assumed_ongoing" for state in regions.values())


def annotated_region_counts(recorded: object, context: dict):
    """Add a distinct context column, never mutate the recorded-start count."""
    result = recorded.copy()
    result["standing_context"] = result["region"].map(
        lambda region: ongoing_count(context, region)
    ).astype(int)
    return result
