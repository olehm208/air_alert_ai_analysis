# AirAlert dashboard v3 — changes since v2

- Four fixed chronological, disjoint 28-day test folds, each preceded by its own 14-day validation period; model selection is performed only on that validation data.
- Added four baselines: training-only overall historical frequency, training-only hour-of-day frequency, and adaptive trailing 7-day and 30-day frequencies. Trailing rates use only labels whose *entire* forecast window has finished by prediction time.
- Added a five-way Brier/AP/ROC comparison, four-fold backtest bar chart and a four-fold scores table.
- Separated actual binary outcomes into their own chart; removed the visually confusing overlay of 0/1 outcomes on predicted probability lines.
- Added independent CSV exports for newest predictions and all fold scores.
- Added tests proving earlier folds and historical rolling baseline values cannot be changed by future alert starts. Fixed no-history prior fallback.
- The old source-separated analytics, region selection, CSV exports and safety/data-quality limitations remain.

## Reproduced finding

For Kyiv City, a six-hour forecast on the supplied October 2026 historical exports, the trailing 30-day frequency baseline beat ML in every fold of both sources. The source-specific holdout Brier scores in the latest fold were volunteer ML **0.177** vs trailing-30d **0.173**, and official ML **0.318** vs trailing-30d **0.246**.

This does not establish a real-world alert prediction capability. It measures prediction of **newly recorded alert starts**, using historical CSV exports with no immutable publication/ingestion timestamps. Use official warnings for safety.
