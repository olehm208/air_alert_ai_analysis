"""AirAlert | two-tab Streamlit dashboard.

Run: streamlit run app.py
Historical analytics and research-only backtesting; not live warning predictions.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    aggregate_counts,
    available_date_range,
    confirmed_durations,
    count_by_region,
    daily_counts,
    filter_alerts,
    hourly_weekday_heatmap,
    source_level_monthly,
)
from data_loader import DEFAULT_DATA_DIR, FILES, REGIONS, load_alerts, region_coverage
from forecasting import BASELINES, HORIZONS, InsufficientDataError, train_and_evaluate

st.set_page_config(
    page_title="AirAlert | Ukrainian Air Raid Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 16px 0 12px;}
.hero-kicker {font-size: .83rem; font-weight: 600; letter-spacing: .14em;
    color: #93ACF8; text-transform: uppercase;}
.hero h1 {margin: .3rem 0 .2rem; font-size: 2.25rem; letter-spacing: -.03em;}
.hero p {color: #9EACC4; margin: .4rem 0 .1rem;}
.note {color: #A9B7D2; font-size: .87rem;}
section[data-testid="stMetric"] {border:1px solid #2B3850; border-radius:14px;
    background: #151D2B; padding: 16px 18px;}
.stTabs [data-baseweb="tab-list"] {gap: 8px; margin-top: 3px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 11px 20px;}
div[data-testid="stAlert"] {border-radius: 11px;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(os.environ.get("AIRALERT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
COLORS = {"volunteer": "#62B6FF", "official": "#C4A0FF"}
SOURCE_NAMES = {"volunteer": "Volunteer", "official": "Official"}


@st.cache_data(show_spinner="Loading air alert history...")
def get_dataset(source: str, directory: str, file_mtime_ns: int) -> pd.DataFrame:
    """The modification timestamp invalidates cache after a regenerated CSV."""
    return load_alerts(source, data_dir=directory)


def source_data(source: str) -> pd.DataFrame | None:
    file = DATA_DIR / FILES[source]
    if not file.exists():
        st.error(f"Missing dataset: `{file}`")
        st.caption("Put the CSV in dataset_repo/.../datasets or set AIRALERT_DATA_DIR.")
        return None
    try:
        return get_dataset(source, str(DATA_DIR), file.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load {source} CSV: {exc}")
        return None


def source_picker(key: str) -> str:
    label = st.segmented_control(
        "Data source", ["Volunteer", "Official"], default="Volunteer", key=key
    )
    return str(label or "Volunteer").lower()


def chart_defaults(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=22, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CFDAEE"),
        xaxis=dict(gridcolor="#263348", zeroline=False),
        yaxis=dict(gridcolor="#263348", zeroline=False),
        hoverlabel=dict(bgcolor="#1A263A"),
    )
    return fig


def data_quality_notice(source: str, rows: pd.DataFrame) -> None:
    earliest, latest = available_date_range(rows)
    st.caption(
        f"Data window: **{earliest.isoformat()} – {latest.isoformat()}** (Kyiv calendar dates). "
        "These files are historical exports, not live alert-status feeds."
    )
    if source == "official":
        st.info(
            "Official dataset: exact duplicate rows are removed. Oblast, raion and "
            "hromada notifications can overlap, so record counts are **not** "
            "necessarily independent air raid events. This export may be older "
            "than the volunteer export."
        )
    else:
        st.info(
            "Volunteer dataset: some ending times are estimated (30 minutes). "
            "Duration charts below exclude these estimates. Counts represent "
            "**reported alert starts**, not confirmed attacks."
        )


def analytics_tab() -> None:
    st.subheader("Explore historical alert records")
    source = source_picker("analytics_source")
    df = source_data(source)
    if df is None or df.empty:
        return
    data_quality_notice(source, df)

    coverage = region_coverage(df)
    by_region = dict(zip(coverage["region"], coverage["coverage"]))
    min_date, max_date = available_date_range(df)
    preferred_start = max(min_date, max_date - timedelta(days=89))

    control1, control2, control3 = st.columns([1.2, 2, 1])
    with control1:
        names = ["All regions"] + sorted(set(REGIONS) | set(df["region"].dropna()))
        choice = st.selectbox("Region", names, key=f"analytics_region_{source}")
    with control2:
        dates = st.date_input(
            "Start dates (Kyiv time)",
            value=(preferred_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"analytics_dates_{source}",
        )
    with control3:
        if source == "official":
            level_choice = st.selectbox(
                "Administrative level",
                ["All levels", "oblast", "raion", "hromada"],
                key="official_level",
            )
        else:
            level_choice = "All levels"
            st.caption("Volunteer data is oblast-level reporting.")

    if not isinstance(dates, (list, tuple)) or len(dates) != 2:
        st.info("Select both a start date and an end date.")
        return
    start, end = dates
    region = None if choice == "All regions" else choice
    level = None if level_choice == "All levels" else level_choice
    filtered = filter_alerts(df, start, end, region=region, level=level)

    if region and by_region.get(region) != "History available":
        st.warning(
            f"**{region}: {by_region.get(region, 'Unknown coverage')}**. "
            "An absent or sparse record history must not be interpreted as zero risk. "
            "Future cancellations and new starts will appear when present in new CSV updates."
        )

    durations = confirmed_durations(filtered)
    represented = filtered["region"].nunique()
    estimated = int(filtered["estimated_end"].sum())
    days = (end - start).days + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded starts", f"{len(filtered):,}")
    m2.metric("Regions represented", f"{represented} / {len(REGIONS)}")
    m3.metric("Starts / calendar day", f"{len(filtered) / days:,.1f}")
    m4.metric("Median confirmed duration", f"{durations.median():.0f} min" if len(durations) else "N/A")
    st.caption(
        f"Window: {days:,} calendar days · Confirmed-duration rows: {len(durations):,} "
        f"· Estimated endings: {estimated:,} · Missing endings: "
        f"{int(filtered['end_unrecorded'].sum()):,}. "
        "Rates are descriptive counts, not probability forecasts."
    )

    if filtered.empty:
        st.warning("No recorded starts match these filters. This does not mean there was no threat.")
        return

    coltitle, colfrequency = st.columns([3, 1], vertical_alignment="bottom")
    with coltitle:
        st.markdown("#### Alert starts over time")
    with colfrequency:
        cadence = st.selectbox("Aggregate by", ["Daily", "Weekly", "Monthly"], key="analytics_cadence")

    timeline = aggregate_counts(daily_counts(filtered, start, end), cadence)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["date"], y=timeline["records"],
        mode="lines", name="Recorded starts", line=dict(color=COLORS[source], width=2.5),
        fill="tozeroy", fillcolor="rgba(100,150,255,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>Starts: %{y:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=None, yaxis_title="Records", showlegend=False)
    st.plotly_chart(chart_defaults(fig, 370), use_container_width=True)
    st.caption(
        "Dates with zero plotted starts mean no starts are recorded in this export "
        "during that date; they do not establish safety or complete reporting."
    )

    col_left, col_right = st.columns([1.25, 1], gap="large")
    with col_left:
        st.markdown("#### Time-of-week distribution")
        heat = hourly_weekday_heatmap(filtered)
        fig = go.Figure(go.Heatmap(
            z=heat.to_numpy(), x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="Blues", hovertemplate="%{y}, %{x}:00 Kyiv<br>%{z:,} starts<extra></extra>",
            colorbar=dict(title="Starts"),
        ))
        fig.update_layout(xaxis_title="Hour (Europe/Kyiv)", yaxis_title=None)
        st.plotly_chart(chart_defaults(fig, 355), use_container_width=True)
    with col_right:
        st.markdown("#### Confirmed alert durations")
        if durations.empty:
            st.info("No confirmed finished durations in the selected data.")
        else:
            cutoff = float(durations.quantile(0.99))
            visible = durations.loc[durations <= cutoff]
            fig = go.Figure(go.Histogram(
                x=visible, nbinsx=35,
                marker_color=COLORS[source], opacity=0.88,
                hovertemplate="Duration: %{x:.0f} min<br>Records: %{y:,}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Duration in minutes", yaxis_title="Records", bargap=0.06)
            st.plotly_chart(chart_defaults(fig, 355), use_container_width=True)
            st.caption(f"99th-percentile visual cutoff: {cutoff:,.0f} min; unusually long durations are excluded from the plot, not the CSV.")

    if region is None:
        st.markdown("#### Regional distribution")
        counts = count_by_region(filtered)
        visible_regions = counts.sort_values("records", ascending=True)
        fig = px.bar(
            visible_regions, x="records", y="region", orientation="h",
            color_discrete_sequence=[COLORS[source]],
            labels={"records": "Recorded starts", "region": "Region"},
        )
        fig.update_layout(showlegend=False, yaxis_title=None)
        st.plotly_chart(chart_defaults(fig, 650), use_container_width=True)
        st.caption("Regional record counts reflect reporting coverage and granularity. They are not directly comparable risk scores.")

    if source == "official":
        st.markdown("#### Changes in official reporting granularity")
        monthly = source_level_monthly(filtered)
        if not monthly.empty:
            fig = px.bar(
                monthly, x="month", y="records", color="level",
                barmode="stack", color_discrete_map={
                    "oblast": "#66B4FF", "raion": "#A78BFA", "hromada": "#69D0AE"
                },
                labels={"month": "Month", "records": "Records", "level": "Level"},
            )
            st.plotly_chart(chart_defaults(fig, 360), use_container_width=True)
        st.caption("Changes in district/community reporting can make historical total-record comparisons misleading.")

    with st.expander("View filtered records and export CSV"):
        st.dataframe(filtered.tail(250), use_container_width=True, hide_index=True)
        st.caption("Showing the last 250 filtered records; download includes all filtered rows.")
        st.download_button(
            "Download filtered observations", data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{source}_alerts_filtered.csv", mime="text/csv",
            key=f"download_{source}",
        )


@st.cache_data(show_spinner="Fitting and backtesting source-specific models…", ttl=3600)
def cached_forecast(
    source: str, region: str, horizon: int, directory: str, file_mtime_ns: int,
):
    """Cache expensive work per source, region, horizon and CSV modification."""
    alerts = get_dataset(source, directory, file_mtime_ns)
    return train_and_evaluate(alerts, region, horizon)


def pretty_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "N/A (one-class test)"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def predictions_tab() -> None:
    st.subheader("Experimental alert-start forecasting")
    source = source_picker("predictions_source")
    df = source_data(source)
    if df is None or df.empty:
        return

    coverage = region_coverage(df)
    coverage_status = dict(zip(coverage["region"], coverage["coverage"]))
    all_regions = sorted(set(REGIONS) | set(df["region"].dropna()))
    a, b = st.columns([2, 1])
    with a:
        region = st.selectbox(
            "Region", all_regions,
            index=all_regions.index("Kyiv City") if "Kyiv City" in all_regions else 0,
            key=f"prediction_region_{source}",
        )
    with b:
        horizon = st.selectbox(
            "Forecast horizon", HORIZONS, index=2,
            format_func=lambda h: f"Next {h} hours",
            key=f"prediction_horizon_{source}",
        )

    last_record = df["started_at"].max()
    observation_end = last_record.floor("h")
    st.caption(
        f"Source: **{SOURCE_NAMES[source]}** · Latest recorded start: "
        f"**{last_record.tz_convert('Europe/Kyiv'):%d %b %Y, %H:%M} Kyiv** · "
        f"Last complete source-hour boundary: "
        f"**{observation_end.tz_convert('Europe/Kyiv'):%d %b %Y, %H:%M} Kyiv**"
    )
    if source == "official":
        st.warning(
            "This official CSV may be stale: no starts are recorded later than "
            f"{last_record.tz_convert('Europe/Kyiv'):%d %b %Y}. "
            "Predictions here are dated historical replays, **not current forecasts**."
        )
    if coverage_status.get(region) != "History available":
        st.warning(
            f"{region}: **{coverage_status.get(region, 'unknown coverage')}**. "
            "No record does not mean no alert. A probability may be unavailable."
        )

    st.info(
        "**Target:** at least one *new recorded alert start* anywhere in the "
        f"selected oblast/city during [{horizon}-hour forecast origin, forecast end). "
        "Official data uses oblast, raion and hromada notices, deduplicated only by "
        "exact record; Volunteer uses oblast-level starts. **These source probabilities "
        "have different meanings and must not be compared as threat levels.**"
    )
    st.caption(
        "Input features contain only starts strictly before the forecast hour; "
        "no finishing times, retrospective durations, news, or future event counts. "
        "Four 28-day walk-forward folds are evaluated. Each fold selects its ML algorithm "
        "using an earlier 14-day validation window, then scores an untouched test period."
    )

    file = DATA_DIR / FILES[source]
    signature = (source, region, horizon, str(DATA_DIR), file.stat().st_mtime_ns)
    if st.button("Train, backtest and show forecast", type="primary", key="fit_forecast"):
        st.session_state["forecast_signature"] = signature
    if st.session_state.get("forecast_signature") != signature:
        st.caption("Select a source, region and horizon, then press the button to run the experiment.")
        return

    try:
        report = cached_forecast(*signature)
    except InsufficientDataError as exc:
        st.error(f"Insufficient data — {exc}")
        st.caption(
            "This source/region remains available in Analytics. "
            "We deliberately do not substitute a fabricated 0% probability."
        )
        return
    except (ValueError, RuntimeError) as exc:
        st.error(f"Forecast could not be computed: {exc}")
        return

    st.markdown("#### Historical snapshot forecast")
    st.caption(
        f"**As of {report.origin.tz_convert('Europe/Kyiv'):%d %b %Y, %H:%M} Kyiv** "
        f"(UTC: {report.origin:%Y-%m-%d %H:%M}). Not a live estimate. "
        "This snapshot model was separately refitted using only labels completed by "
        "the snapshot time; it was never used for the held-out scores."
    )
    latest_roll = report.all_baseline_metrics["rolling_30d"]["brier"]
    all_briers = {key: score["brier"] for key, score in report.all_baseline_metrics.items()}
    best_baseline_key = min(all_briers, key=all_briers.get)
    best_baseline_brier = all_briers[best_baseline_key]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"New recorded start, next {horizon}h", f"{report.probability:.1%}")
    m2.metric("Latest test Brier · ML ↓", pretty_metric(report.metrics["brier"]))
    m3.metric("Latest test Brier · rolling 30d ↓", pretty_metric(latest_roll))
    m4.metric("Test positive rate", pretty_metric(report.metrics["positive_rate"], percent=True))

    if report.metrics["brier"] < best_baseline_brier:
        st.success(
            "The ML model scored better than all four baselines on the **latest held-out "
            "period** (Brier score). This is a historical result, not proof of real-world reliability."
        )
    else:
        st.warning(
            "The ML model did **not** beat every baseline on the latest held-out period. "
            f"The lowest-error baseline was **{BASELINES[best_baseline_key]}** "
            f"(Brier {best_baseline_brier:.3f} vs ML {report.metrics['brier']:.3f}). "
            "The best baseline is identified *after* evaluation for comparison, "
            "not used to select or tune the ML algorithm."
        )

    st.markdown("#### Latest held-out period · forecast probabilities")
    test = report.test
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["model_probability"], mode="lines",
        name=f"ML · {report.model_name}", line=dict(color=COLORS[source], width=2.8),
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["rolling_30d_probability"], mode="lines",
        name="Trailing 30 days", line=dict(color="#63D2B1", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["rolling_7d_probability"], mode="lines",
        name="Trailing 7 days", line=dict(color="#FFD074", width=1.6, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["hour_of_day_probability"], mode="lines",
        name="Historical hour-of-day", line=dict(color="#B6A5D5", width=1.1, dash="dash"),
        opacity=0.7,
    ))
    fig.update_layout(
        yaxis=dict(title="Predicted probability", range=[0, 1], tickformat=".0%"),
        xaxis_title="Forecast origin · Kyiv time", legend=dict(orientation="h", y=-0.22),
    )
    st.plotly_chart(chart_defaults(fig, 380), use_container_width=True)

    st.markdown("##### What actually happened in each forecast window")
    observed = go.Figure()
    observed.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["actual"], mode="markers",
        name="Recorded alert start (1=yes)",
        marker=dict(
            color=["#66DAB8" if x == 1 else "#8B96AA" for x in test["actual"]],
            size=8, opacity=0.85,
        ),
        hovertemplate="%{x}<br>At least one new recorded start: %{y}<extra></extra>",
    ))
    observed.update_layout(
        yaxis=dict(title="Observed label", range=[-0.22, 1.22], tickvals=[0, 1],
                   ticktext=["No start", "Start"]),
        xaxis_title="Forecast origin · Kyiv time", showlegend=False,
    )
    st.plotly_chart(chart_defaults(observed, 210), use_container_width=True)
    st.caption(
        f"Each dot is one non-overlapping {horizon}-hour window. "
        "A dot marked 'No start' means no *recorded* new start in that window, "
        "not that the region was safe or free of an active alert. "
        "Rolling baselines update using prior labels only after their full forecast horizons end."
    )

    st.markdown("#### Latest test: all five approaches")
    comparison_rows = [
        {"Approach": f"ML · {report.model_name}", "Brier ↓": report.metrics["brier"],
         "Average precision ↑": report.metrics["average_precision"],
         "ROC AUC ↑": report.metrics["roc_auc"]},
    ]
    for key, label in BASELINES.items():
        scores = report.all_baseline_metrics[key]
        comparison_rows.append({
            "Approach": label, "Brier ↓": scores["brier"],
            "Average precision ↑": scores["average_precision"],
            "ROC AUC ↑": scores["roc_auc"],
        })
    st.dataframe(
        pd.DataFrame(comparison_rows).style.format(
            {"Brier ↓": "{:.3f}", "Average precision ↑": "{:.3f}", "ROC AUC ↑": "{:.3f}"},
            na_rep="N/A",
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Lower Brier is better. The constant and hour-of-day baselines use only "
        "pre-test training labels. Rolling baselines use a trailing 7/30-day "
        "window of labels *matured by each forecast time*, including earlier "
        "test-period outcomes once they become historically observable. "
        "AUC/AP can be unavailable if a test period has one outcome class."
    )

    st.markdown("#### Walk-forward evaluation · four disjoint 28-day folds")
    folds = report.walk_forward.copy()
    long = folds.melt(
        id_vars=["fold_start", "fold_end"],
        value_vars=["brier_model"] + [f"brier_{key}" for key in BASELINES],
        var_name="approach", value_name="Brier score",
    )
    display_names = {"brier_model": "ML"}
    display_names.update({f"brier_{key}": label for key, label in BASELINES.items()})
    long["approach"] = long["approach"].map(display_names)
    long["period"] = long["fold_start"].dt.strftime("%d %b") + " – " + long["fold_end"].dt.strftime("%d %b %Y")
    fold_fig = px.bar(
        long, x="period", y="Brier score", color="approach", barmode="group",
        category_orders={"approach": ["ML", *BASELINES.values()]},
        color_discrete_map={
            "ML": COLORS[source], "Past overall frequency": "#8998B7",
            "Past hour-of-day frequency": "#B6A5D5",
            "Trailing 7-day frequency": "#FFD074",
            "Trailing 30-day frequency": "#63D2B1",
        },
    )
    fold_fig.update_layout(xaxis_title="Test interval", legend_title=None,
                           legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(chart_defaults(fold_fig, 400), use_container_width=True)
    st.caption(
        "For each fold, training ends before the validation window, a model is "
        "selected using validation Brier, and the next 28 days are evaluated. "
        "No test-fold labels enter that fold's model. Earlier fold outcomes can "
        "legitimately enter later folds after their forecast windows have ended."
    )

    score_cols = ["brier_model"] + [f"brier_{key}" for key in BASELINES]
    mean_briers = folds[score_cols].mean().rename(index=display_names)
    best_by_mean = mean_briers.idxmin()
    ml_wins = int((folds["brier_model"] < folds["brier_rolling_30d"]).sum())
    a, b, c = st.columns(3)
    a.metric("Walk-forward folds", str(len(folds)))
    b.metric("ML beats rolling 30d", f"{ml_wins} / {len(folds)}")
    c.metric("Lowest mean fold Brier", best_by_mean)
    st.dataframe(
        folds.assign(
            fold_start=folds["fold_start"].dt.strftime("%Y-%m-%d"),
            fold_end=folds["fold_end"].dt.strftime("%Y-%m-%d"),
        ).rename(columns={"fold_start": "Test begins (UTC)",
                          "fold_end": "Test ends (UTC)",
                          "selected_model": "Selected ML model",
                          "positive_rate": "Positive rate"}),
        hide_index=True, use_container_width=True,
    )

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("#### Evaluation configuration")
        st.write(f"Latest-fold selected ML algorithm: **{report.model_name}**")
        st.write(f"Train origins: **{report.train_samples:,}** · Validation: **{report.validation_samples:,}** · Test: **{report.test_samples:,}**")
        st.write(f"Recent recorded starts (240-day input window): **{report.recent_starts:,}**")
        st.caption(
            "The forecast at the end of the file is a historical replay. "
            "Selection is by validation Brier in each fold, never by test results. "
            "This analysis is not a model of military activity or an alert-status service."
        )
    with right:
        st.markdown("#### Latest test reliability")
        sample = test.copy()
        sample["bucket"] = pd.cut(
            sample["model_probability"], bins=[0, .2, .4, .6, .8, 1.00001],
            labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
            include_lowest=True,
        )
        reliability = sample.groupby("bucket", observed=False).agg(
            predicted=("model_probability", "mean"),
            observed=("actual", "mean"), count=("actual", "size"),
        ).reset_index()
        reliability = reliability[reliability["count"] > 0]
        if len(reliability):
            cal = go.Figure()
            cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(dash="dash", color="#A0AEC0"),
            ))
            cal.add_trace(go.Scatter(
                x=reliability["predicted"], y=reliability["observed"],
                mode="markers+lines", name="Held-out ML calibration",
                customdata=reliability["count"],
                marker=dict(size=10, color=COLORS[source]),
                hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Samples %{customdata}<extra></extra>",
            ))
            cal.update_layout(
                xaxis=dict(title="Mean predicted probability", range=[0, 1]),
                yaxis=dict(title="Observed positive rate", range=[0, 1]),
            )
            st.plotly_chart(chart_defaults(cal, 310), use_container_width=True)
            st.caption("Only populated prediction bands are shown; groups may be small.")

    with st.expander("Exact chronological boundaries and CSV downloads"):
        st.write(f"Latest-fold training labels mature no later than **{report.periods['train_labels_end']}**")
        st.write(f"Latest-fold validation labels mature no later than **{report.periods['validation_labels_end']}**")
        st.write(f"Latest-fold test labels mature no later than **{report.periods['test_labels_end']}**")
        st.write(f"Validation Brier by algorithm: {report.validation_scores}")
        st.dataframe(test.tail(60), hide_index=True, use_container_width=True)
        st.download_button(
            "Download latest held-out predictions",
            data=test.to_csv(index=False).encode(),
            file_name=f"{source}_{region.replace(' ', '_')}_{horizon}h_test.csv",
            mime="text/csv", key="forecast_export",
        )
        st.download_button(
            "Download walk-forward fold results",
            data=folds.to_csv(index=False).encode(),
            file_name=f"{source}_{region.replace(' ', '_')}_{horizon}h_walkforward.csv",
            mime="text/csv", key="fold_export",
        )
    st.warning(
        "**Safety & data limits:** This is a research demonstration of *recorded alert starts*, "
        "not a detector of incoming threats. Missing rows, delayed/edited source posts and "
        "active or permanently ongoing alerts cannot be interpreted as safety. "
        "The exports lack publication/ingestion timestamps, so historical availability "
        "cannot be proven for retrospectively backfilled messages. "
        "Always follow official air-raid warnings."
    )


st.markdown("""
<div class="hero">
    <div class="hero-kicker">Ukraine · time series research</div>
    <h1>AirAlert Analytics</h1>
    <p>Historical air raid alerts, source-aware analysis, and evaluated forecasting experiments.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "**Research tool only.** Historical records and future experimental forecasts "
    "must never replace official air raid alerts or emergency guidance.", icon="⚠️"
)

analytics, predictions = st.tabs(["📈  Analytics", "🔮  Predictions"])
with analytics:
    analytics_tab()
with predictions:
    predictions_tab()

st.divider()
st.caption(
    "AirAlert Analytics · All plotted timestamps use Kyiv local time; preprocessing "
    "stores UTC timestamps. Volunteer and official sources are never silently mixed."
)
