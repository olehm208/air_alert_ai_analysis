"""Manual standing alerts: never invent a historical start or a model label."""
import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from standing_alerts import (
    DEFAULT_PATH,
    load_standing_context,
    ongoing_count,
    annotated_region_counts,
)


def test_default_one_continuous_alert_per_region():
    context = load_standing_context()
    assert ongoing_count(context) == 3
    assert ongoing_count(context, "Luhanska oblast") == 1
    assert ongoing_count(context, "Crimea (Autonomous Republic)") == 1
    assert ongoing_count(context, "Sevastopol City") == 1
    assert ongoing_count(context, "Kyiv City") == 0
    assert context["as_of_date"] == "2026-10-10"


def test_manual_cancellation_removes_only_standing_flag(tmp_path: Path):
    config = json.loads(DEFAULT_PATH.read_text())
    config["regions"]["Sevastopol City"] = "ended"
    config["as_of_date"] = "2026-10-11"
    path = tmp_path / "standing_alerts.json"
    path.write_text(json.dumps(config))
    updated = load_standing_context(path)
    assert ongoing_count(updated) == 2
    assert ongoing_count(updated, "Sevastopol City") == 0
    assert ongoing_count(updated, "Luhanska oblast") == 1


def test_standing_counts_are_separate_from_csv_recorded_starts():
    recorded = pd.DataFrame({
        "region": ["Crimea (Autonomous Republic)", "Sevastopol City", "Luhanska oblast", "Kyiv City"],
        "records": [0, 0, 0, 42],
    })
    with_context = annotated_region_counts(recorded, load_standing_context())
    assert with_context["records"].tolist() == [0, 0, 0, 42]
    assert with_context["standing_context"].tolist() == [1, 1, 1, 0]
    assert recorded.columns.tolist() == ["region", "records"]


def test_rejects_invalid_context_status(tmp_path: Path):
    config = json.loads(DEFAULT_PATH.read_text())
    config["regions"]["Luhanska oblast"] = "definitely_safe"
    path = tmp_path / "standing_alerts.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_standing_context(path)


def test_sidebar_nav_buttons_have_no_hover_tooltips():
    """Regression: a Streamlit help tooltip obscured the next icon button."""
    tree = ast.parse((Path(__file__).resolve().parent / "app.py").read_text())
    sidebar = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sidebar_nav")
    nav_button = next(
        n for n in ast.walk(sidebar)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "button"
        and any(k.arg == "key" and isinstance(k.value, ast.JoinedStr) for k in n.keywords)
    )
    assert not any(k.arg == "help" for k in nav_button.keywords)
