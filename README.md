# AirAlert Analytics

A Python/Streamlit dashboard with two tabs:

- **Analytics:** real, interactive statistics from separately selectable **Volunteer** or **Official** CSV exports, with region/date filters, charts, and CSV export.
- **Predictions:** UI prepared for **two separate forecasting experiments**, but **no fabricated probabilities**.
  
> **Important:** This is a retrospective research project, **not** a real-time alert monitoring service. No plotted count or future prediction should be used in safety decisions. A missing record does **not** imply that an area is safe.

## Project layout

```text
AI_Proj/
├── app.py
├── analytics.py
├── data_loader.py
├── inspect_data.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── test_analytics.py
└── test_data_loader.py
```

## Start

From inside your existing virtual environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

Streamlit normally opens your browser at `http://localhost:8501`.

If you want to keep the datasets in a different directory, point `AIRALERT_DATA_DIR` at the directory holding the two CSV files:

```bash
AIRALERT_DATA_DIR=/path/to/datasets python -m streamlit run app.py
```

Dataset regeneration updates the dashboard cache automatically based on the CSV's last-modified time. Browser tab selection can be changed independently between Volunteer and Official.

## What each chart means

- **Recorded starts:** number of dataset rows whose start timestamp falls inside the chosen dates, after exact duplicate removal.
- **Daily/weekly/monthly trend:** dates with zero starts mean **no starts recorded by this source**, not proof that all alerts were inactive.
- **Time-of-week heatmap:** distribution of *recorded start* times in Europe/Kyiv; not probability, risk, or a forecast.
- **Confirmed durations:** excludes volunteer rows with `naive=True` estimated finish times, plus rows without an end timestamp.
- **Regional distribution:** geographic *record* frequency, not independent threat frequency. Official notifications at oblast, raion and hromada levels can overlap.
- **Official reporting mix:** shows how records by administrative reporting level change over time.

All 27 administrative areas in the current data loader are selectable, including Crimea, Sevastopol, and Luhansk. Missing and sparse data are called out, rather than interpreted as zero risk. New observations will appear after refreshing the exported CSVs. **The dashboard cannot see cancellations that its upstream data generator does not record**, so you should never interpret this dashboard as current alert status.


## Privacy

Do not commit the external dataset generator directory, `.env` secrets or Telegram sessions. The dashboard does **not** need Telethon or account credentials to analyze already exported CSV files.
