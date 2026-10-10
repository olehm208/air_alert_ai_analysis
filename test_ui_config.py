"""Native Streamlit themes should provide readable UI in both appearances."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent


def _relative_luminance(rgb: str) -> float:
    v = [int(rgb[i:i+2], 16) / 255 for i in (1, 3, 5)]
    linear = [c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in v]
    return .2126*linear[0] + .7152*linear[1] + .0722*linear[2]


def test_both_native_appearances_have_readable_text():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    theme = cfg["theme"]
    for mode in ("dark", "light"):
        for palette in (theme[mode], theme[mode]["sidebar"]):
            fg = _relative_luminance(palette["textColor"])
            bg = _relative_luminance(palette["backgroundColor"])
            ratio = (max(fg, bg)+.05)/(min(fg, bg)+.05)
            assert ratio >= 7.0, (mode, ratio)


def test_page_does_not_force_light_to_use_dark_css_text():
    source = (ROOT / "app.py").read_text()
    assert 'native_mode()' in source
    assert 'st.context.theme.type' in source
    assert 'st.session_state.ui_theme' not in source


def test_every_chart_accent_is_resolvable_in_both_native_themes():
    """Regression: v5's dark dictionary recursively called chart_accent itself."""
    import ast
    root = ast.parse((ROOT / "app.py").read_text())
    func = next(
        node for node in root.body
        if isinstance(node, ast.FunctionDef) and node.name == "chart_accent"
    )
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    names = (
        "source_volunteer", "source_official", "rolling_30", "rolling_7",
        "hour", "overall", "observed_yes", "observed_no",
        "oblast", "raion", "hromada", "reference",
    )
    for mode in ("dark", "light"):
        namespace = {"native_mode": lambda: mode}
        exec(compile(module, "app.py", "exec"), namespace)
        for name in names:
            color = namespace["chart_accent"](name)
            assert color.startswith("#") and len(color) == 7, (mode, name, color)


def test_nav_uses_locked_rail_with_app_controlled_expansion():
    source = (ROOT / "app.py").read_text()
    assert 'initial_sidebar_state="locked"' in source
    assert 'st.session_state.sidebar_expanded = not expanded' in source
    assert 'key="toggle_sidebar_width"' in source
    assert 'key=f"nav_{page.lower()}"' in source
    assert 'st.session_state.amoled' not in source


def test_dark_is_amoled_black_and_light_stays_light():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["theme"]["dark"]["backgroundColor"] == "#000000"
    assert config["theme"]["dark"]["sidebar"]["backgroundColor"] == "#000000"
    assert _relative_luminance(config["theme"]["light"]["backgroundColor"]) > 0.8
