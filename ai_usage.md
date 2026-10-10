# AI usage and engineering accountability

## Statement

**ChatGPT was the primary engineering assistant for this project.** The application was developed iteratively with AI-generated Python, test and documentation changes under human direction and review. The developer was responsible for choosing the problem, deciding what to build, collecting the historical data, protecting Telegram credentials, executing and testing the application, inspecting outputs, and deciding which changes to accept.

**This is a description of the actual workflow**, not a claim that an external hosted language-model API is required at runtime. The dashboard runs locally with Python libraries; scikit-learn models provide the forecasting experiments. OpenAI/Gemini/Anthropic API calls are **not** needed by the finished application.

## Division of responsibilities

| Area | Human contribution | AI-assisted contribution |
|---|---|---|
| Problem definition | Selected Ukrainian air-raid-alert time series and the deadline-constrained mini-project scope | Helped convert the broad task into an empirical forecasting research question |
| Data acquisition | Obtained Telegram user access privately; generated and supplied both CSV exports | Helped troubleshoot Telethon login and interpret dataset structure |
| Architecture | Chose a web dashboard, separate source modes, analytics/prediction views, and special treatment of sparse regions | Generated Streamlit modules and data-loading / chart architecture |
| Modeling | Requested objective forecasts, stronger benchmarks and evaluations; checked results visually | Implemented lag-only features, ML candidates, baselines, chronological splits and scoring |
| UI and accessibility | Reported responsiveness problems, light-mode contrast, tooltips, and navigation bugs | Implemented native themes, compact navigation, source popover and tooltip corrections |
| Reliability | Executed code, provided tracebacks, and tested in the browser | Wrote and maintained pytest regressions, including time-cutoff tests |
| Communication | Selected final features and reviewed what outputs mean | Drafted README, methodology, warnings and reproducibility guidance |

## Development milestones and corrections

1. **First loader:** built a source-aware parser, removed exact duplicates, kept estimated volunteer endings identifiable, retained all 27 regions and added tests.
2. **Initial dashboard:** added interactive descriptive analytics and separate prediction controls using Streamlit and Plotly.
3. **Initial forecasting:** implemented logistic regression / gradient boosting and chronological holdout evaluation. The first benchmark was too weak, so its apparently positive results were treated skeptically.
4. **More credible evaluation:** added pre-test global/hourly baselines, trailing 7-day and 30-day online baselines, four 28-day walk-forward folds and reliability plots. The stronger 30-day baseline *beat ML* in the latest **Kyiv City, 6-hour** volunteer snapshot; the result was kept and reported rather than suppressed.
5. **UI bug fixing:** the developer reported a `RecursionError` from a recursive color dictionary; AI rewrote the palette helper and added regression tests. The developer also found light-theme contrast and hover-tooltip problems; the UI was iterated based on local screenshots.
6. **Standing alerts:** the developer requested one assumed ongoing alert for Luhansk, Crimea and Sevastopol. AI implemented independent manual annotations so those assumptions cannot contaminate recorded-start counts or forecasting labels.
7. **Submission pass:** AI prepared editable installation metadata (`setup.py`), method documentation, an editable About Me page, safer `.gitignore` rules and the final README. The developer must review the personal profile and run release tests on the submission machine.

## Model-selection and data-leakage review

AI assistance was used to design and test invariants: observations and features precede their forecast origin; each label's window must fully elapse before training or rolling-baseline inclusion; validation chooses the algorithm; test labels never select/train that tested model; earlier test-period outcomes become permissible only *after they have matured* in online-updated baselines.

There is a **remaining data-origin limitation**: the retrospective CSV exports do not preserve each message's original publication/ingestion timestamp, so we cannot prove a late edit or backfill was available at its simulated forecast time. The app does **not** claim this problem is solved. A more rigorous future project would store append-only point-in-time snapshots.

## What AI did not do

- It did **not** verify live alert states, deploy an operational warning service, or predict actual attacks.
- It did **not** need the developer's Telegram session string or API hash to run the dashboard.
- It did **not** use future ground-truth labels in the training of the model evaluated on a given test fold.
- It did **not** independently verify original Telegram-message availability at every historical forecast timestamp.
