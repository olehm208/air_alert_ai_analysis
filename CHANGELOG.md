## v7 — October 10, 2026

- Removed navigation-button `help=` hover tooltips that obstructed adjacent sidebar buttons.
- Added an editable `standing_alerts.json` manual-context register with **1 assumed ongoing alert** each for Luhansk, Crimea and Sevastopol, and an explicit `ended` state for later verified cancellations.
- Analytics clearly separates CSV-recorded starts from standing-context counts; Predictions display the annotation without using it to fabricate probabilities.
- Added five new regression tests; all 34 tests pass.
- No changes to ML features, targets, scoring, chronological splits or source handling.

# AirAlert Dashboard v6 — fix theme and navigation

- Fixed the **`RecursionError`** in the dark chart palette. Region-specific dark colors are now constant hex values rather than self-recursive `chart_accent()` calls. Added a regression test that calls every palette color in both Light and Dark appearances.
- Native Streamlit **Dark** now uses an AMOLED-black `#000000` page *and sidebar* in `.streamlit/config.toml`. Native Light remains independently styled by Streamlit.
- Removed the application-specific AMOLED toggle. Use **⋮ → Settings → Light/Dark** (the Dark choice is AMOLED).
- Native desktop sidebar **hide** is replaced with a custom **expand/compact** control. Streamlit's `initial_sidebar_state="locked"` keeps the sidebar pinned on desktop; the custom first button switches between an 86px icon rail and a 230px labeled sidebar. Mobile still uses Streamlit's adaptive collapse behavior.
- The global source selector remains in the top-right **⚙️** popover; Streamlit's native menu cannot accept arbitrary Python widgets.
- Removed styling unrelated to layout; the native theme supplies widget/alert colors in both modes. Forecasting and analytics computations are unchanged.
