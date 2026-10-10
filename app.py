"""AirAlert | Streamlit dashboard.

Run: streamlit run app.py
Historical analytics and research-only backtesting; not live warning predictions.
"""
from __future__ import annotations

import os
from datetime import timedelta
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
from standing_alerts import DEFAULT_PATH as STANDING_PATH, load_standing_context, ongoing_count, annotated_region_counts

st.set_page_config(
    page_title="AirAlert | Ukrainian Air Raid Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="locked",
    menu_items={"About": "**AirAlert Analytics** — a historical air-raid alert research dashboard, not an emergency service."},
)

DATA_DIR = Path(os.environ.get("AIRALERT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
SOURCE_NAMES = {"volunteer": "Volunteer", "official": "Official"}
COLORS = {"volunteer": "#418FDC", "official": "#8F70DA"}

if "global_source" not in st.session_state:
    st.session_state.global_source = "volunteer"
if "active_page" not in st.session_state:
    st.session_state.active_page = "Analytics"
if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = False


def native_mode() -> str:
    """Uses Streamlit's *actual* active appearance, not a CSS-only theme guess."""
    try:
        return "light" if st.context.theme.type == "light" else "dark"
    except (AttributeError, KeyError):
        return "dark"


def current_source() -> str:
    return st.session_state.global_source


def current_source_color() -> str:
    return chart_accent(f"source_{current_source()}")


def current_palette() -> dict[str, str]:
    if native_mode() == "light":
        return {
            "text": "#1B2638", "grid": "#DCE4F0", "hover_bg": "#FFFFFF",
            "hover_text": "#172235", "border": "#CBD5E1", "plot_bg": "rgba(0,0,0,0)",
            "fill_rgba": "rgba(65,143,220,0.10)", "heat_scale": "Blues",
        }
    return {
        "text": "#E5ECF6", "grid": "#273347", "hover_bg": "#152237",
        "hover_text": "#F0F5FF", "border": "#34455F", "plot_bg": "rgba(0,0,0,0)",
        "fill_rgba": "rgba(98,182,255,0.10)", "heat_scale": "Blues",
    }


def chart_accent(name: str) -> str:
    """Readable chart colors in both native themes."""
    light = {
        "source_volunteer": "#2366AB", "source_official": "#7053B2",
        "rolling_30": "#137A60", "rolling_7": "#A86200", "hour": "#6E5799",
        "overall": "#526580", "observed_yes": "#087F62", "observed_no": "#6D7A8C",
        "oblast": "#276BB5", "raion": "#7654BC", "hromada": "#10856C",
        "reference": "#7B8797",
    }
    dark = {
        "source_volunteer": "#62B6FF", "source_official": "#C4A0FF",
        "rolling_30": "#63D2B1", "rolling_7": "#FFD074", "hour": "#B6A5D5",
        "overall": "#8998B7", "observed_yes": "#66DAB8", "observed_no": "#8B96AA",
        "oblast": "#66B4FF", "raion": "#A78BFA", "hromada": "#69D0AE",
        "reference": "#A0AEC0",
    }
    return (light if native_mode() == "light" else dark)[name]


def apply_layout_css() -> None:
    """Layout only: native Streamlit themes own all foreground/background colors.

    Desktop navigation is pinned but can switch between compact and expanded
    via our own button. Streamlit retains native mobile collapse behavior.
    The sidebar CSS selectors are presentation-only and scoped to its public
    data-testid attributes, which may require an update with future versions.
    """
    expanded = st.session_state.sidebar_expanded
    width = 230 if expanded else 86
    padding = "0.8rem" if expanded else "0.45rem"
    st.markdown(f"""
    <style>
        .block-container {{max-width: 1490px; padding-top: 1.3rem; padding-bottom: 2.5rem;}}
        .hero {{padding: 3px 0 11px;}}
        .hero-kicker {{font-size: .73rem; font-weight: 700; letter-spacing: .13em;
            opacity: .75; text-transform: uppercase;}}
        .hero h1 {{margin: .25rem 0 .15rem; font-size: clamp(1.75rem, 3vw, 2.4rem);
            letter-spacing: -.03em;}}
        .hero p {{opacity: .74; margin: .35rem 0 0; font-size: .95rem;}}
        [data-testid="stSidebar"][aria-expanded="true"] {{
            width: {width}px !important;
            min-width: {width}px !important;
            max-width: {width}px !important;
            flex-basis: {width}px !important;
        }}
        [data-testid="stSidebarContent"] {{
            padding: 0.6rem {padding} !important;
            width: 100% !important;
        }}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{gap: .35rem;}}
        [data-testid="stSidebar"] button {{
            min-height: 45px;
            font-size: {"0.9rem" if expanded else "1.25rem"};
            padding: .3rem .5rem;
        }}
        [data-testid="stSidebarResizeHandle"] {{display: none !important;}}
        section[data-testid="stMetric"] {{
            border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
            border-radius: 13px; padding: 12px 14px;
        }}
        div[data-testid="stAlert"] {{border-radius: 12px;}}
        @media (max-width: 800px) {{
           .block-container {{padding: 1.5rem 1rem 2rem !important;}}
           .hero p {{font-size: .86rem;}}
        }}
    </style>
    """, unsafe_allow_html=True)

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


def chart_defaults(fig: go.Figure, height: int = 360) -> go.Figure:
    p = current_palette()
    fig.update_layout(
        template="plotly_white" if native_mode() == "light" else "plotly_dark",
        height=height,
        margin=dict(l=8, r=12, t=22, b=12),
        paper_bgcolor=p["plot_bg"], plot_bgcolor=p["plot_bg"],
        font=dict(color=p["text"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=p["hover_bg"], bordercolor=p["border"],
            font=dict(color=p["hover_text"], size=13), namelength=-1,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=p["text"])),
    )
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"], color=p["text"])
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"], color=p["text"])
    fig.update_traces(
        hoverlabel=dict(font=dict(color=p["hover_text"]), bgcolor=p["hover_bg"], bordercolor=p["border"])
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


def render_global_header() -> None:
    title, options = st.columns([7.5, 1], vertical_alignment="top")
    with title:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-kicker">Ukraine · time series research</div>
                <h1>AirAlert Analytics</h1>
                <p>Historical alerts, source-aware analysis and evaluated forecasts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with options:
        with st.popover("⚙️", help="Choose data source", use_container_width=True):
            st.markdown("**Data source**")
            selected = st.radio(
                "Historical data", ["Volunteer", "Official"],
                index=0 if current_source() == "volunteer" else 1,
                label_visibility="collapsed", horizontal=True,
                key="source_in_settings",
            )
            st.session_state.global_source = selected.lower()
            st.caption("Sources stay independent in analytics and predictions.")


@st.cache_data(show_spinner=False)
def cached_standing_context(path: str, file_mtime_ns: int) -> dict:
    """Refresh the annotation when the local JSON is edited."""
    return load_standing_context(path)


def standing_context() -> dict:
    return cached_standing_context(str(STANDING_PATH), STANDING_PATH.stat().st_mtime_ns)


def show_standing_annotation(context: dict, region: str | None = None) -> None:
    """Show manually assumed current context; never treat it as CSV evidence."""
    count = ongoing_count(context, region)
    if count:
        noun = "alert" if count == 1 else "alerts"
        st.info(
            f"**{count} standing {noun} (assumed)** · As of {context['as_of_date']}. "
            "This is a manually maintained project annotation, **not a CSV observation "
            "or live-verified status**. It does not affect recorded starts or forecasts. "
            "Edit `standing_alerts.json` to mark a verified cancellation as `ended`."
        )
    elif region in context['regions']:
        st.info(f"**{region}: standing annotation ended** · Last updated {context['as_of_date']}. "
                "Review the updated historical source separately for any recorded cancellation.")


def sidebar_nav() -> str:
    """An always-visible desktop rail, toggling compact icons ↔ labeled nav."""
    expanded = st.session_state.sidebar_expanded
    with st.sidebar:
        if st.button(
            "◀  Compact" if expanded else "»",
            help="Switch to compact navigation" if expanded else "Expand navigation",
            key="toggle_sidebar_width",
            use_container_width=True,
        ):
            st.session_state.sidebar_expanded = not expanded
            st.rerun()

        for page, icon in (("Analytics", "📈"), ("Predictions", "🔮")):
            label = f"{icon}  {page}" if expanded else icon
            st.button(
                label,
                key=f"nav_{page.lower()}",
                use_container_width=True,
                type="primary" if st.session_state.active_page == page else "secondary",
                on_click=lambda name=page: st.session_state.update(active_page=name),
            )
    return st.session_state.active_page


def analytics_tab(source: str) -> None:
    context = standing_context()
    st.subheader("Explore historical alert records")
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
    show_standing_annotation(context, region)

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

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Recorded starts", f"{len(filtered):,}")
        st.metric("Starts / calendar day", f"{len(filtered) / days:,.1f}")
    with m2:
        st.metric("Regions represented", f"{represented} / {len(REGIONS)}")
        st.metric("Median confirmed duration", f"{durations.median():.0f} min" if len(durations) else "N/A")
    st.caption(
        f"Window: {days:,} calendar days · Confirmed-duration rows: {len(durations):,} "
        f"· Estimated endings: {estimated:,} · Missing endings: "
        f"{int(filtered['end_unrecorded'].sum()):,}. "
        "Rates are descriptive counts, not probability forecasts."
    )

    if filtered.empty:
        st.warning("No recorded starts match these filters. This does not mean there was no threat. "
                   "Standing-alert annotations (if shown) remain independent of the selected dates.")
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
        mode="lines", name="Recorded starts", line=dict(color=current_source_color(), width=2.5),
        fill="tozeroy", fillcolor=current_palette()["fill_rgba"],
        hovertemplate="%{x|%Y-%m-%d}<br>Starts: %{y:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=None, yaxis_title="Records", showlegend=False)
    st.plotly_chart(chart_defaults(fig, 370), use_container_width=True, config={"responsive": True})
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
            colorscale=current_palette()["heat_scale"], hovertemplate="%{y}, %{x}:00 Kyiv<br>%{z:,} starts<extra></extra>",
            colorbar=dict(title="Starts"),
        ))
        fig.update_layout(xaxis_title="Hour (Europe/Kyiv)", yaxis_title=None)
        st.plotly_chart(chart_defaults(fig, 355), use_container_width=True, config={"responsive": True})
    with col_right:
        st.markdown("#### Confirmed alert durations")
        if durations.empty:
            st.info("No confirmed finished durations in the selected data.")
        else:
            cutoff = float(durations.quantile(0.99))
            visible = durations.loc[durations <= cutoff]
            fig = go.Figure(go.Histogram(
                x=visible, nbinsx=35,
                marker_color=current_source_color(), opacity=0.88,
                hovertemplate="Duration: %{x:.0f} min<br>Records: %{y:,}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Duration in minutes", yaxis_title="Records", bargap=0.06)
            st.plotly_chart(chart_defaults(fig, 355), use_container_width=True, config={"responsive": True})
            st.caption(f"99th-percentile visual cutoff: {cutoff:,.0f} min; unusually long durations are excluded from the plot, not the CSV.")

    if region is None:
        st.markdown("#### Regional distribution")
        counts = annotated_region_counts(count_by_region(filtered), context)
        visible_regions = counts.sort_values(["records", "region"], ascending=[True, True])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=visible_regions["records"], y=visible_regions["region"],
            name="Recorded starts in selected dates", orientation="h",
            marker_color=current_source_color(),
            hovertemplate="%{y}<br>CSV-recorded starts: %{x:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=visible_regions["standing_context"], y=visible_regions["region"],
            name="Standing alert (manual context)", orientation="h",
            marker_color=chart_accent("rolling_30"),
            hovertemplate="%{y}<br>Assumed ongoing alerts: %{x:,}<extra></extra>",
        ))
        fig.update_layout(barmode="group", xaxis_title="Count (different definitions)",
                          yaxis_title=None, legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(chart_defaults(fig, 490), use_container_width=True, config={"responsive": True})
        standing = counts.loc[counts["standing_context"].eq(1), ["region", "standing_context"]]
        if not standing.empty:
            st.caption(
                "Separate standing-alert annotations (1 each): "
                + ", ".join(standing["region"].tolist())
                + f". These are manually assumed as of {context['as_of_date']}, regardless of the "
                  "selected start-date range; they are not extra recorded starts."
            )

    if source == "official":
        st.markdown("#### Official reporting levels over time")
        level_counts = source_level_monthly(filtered)
        if not level_counts.empty:
            fig = px.bar(
                level_counts,
                x="month", y="records", color="level",
                barmode="stack", color_discrete_map={
                    "oblast": chart_accent("oblast"), "raion": chart_accent("raion"), "hromada": chart_accent("hromada")
                },
                labels={"month": "Month", "records": "Records", "level": "Level"},
            )
            st.plotly_chart(chart_defaults(fig, 360), use_container_width=True, config={"responsive": True})
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
def cached_forecast(source: str, region: str, horizon: int, directory: str, file_mtime_ns: int):
    alerts = get_dataset(source, directory, file_mtime_ns)
    return train_and_evaluate(alerts, region, horizon)


def pretty_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "N/A (one-class test)"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def predictions_tab(source: str) -> None:
    context = standing_context()
    st.subheader("Experimental alert-start forecasting")
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
    show_standing_annotation(context, region)
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

    m1, m2 = st.columns(2)
    with m1:
        st.metric(f"New recorded start, next {horizon}h", f"{report.probability:.1%}")
        st.metric("Latest test Brier · ML ↓", pretty_metric(report.metrics["brier"]))
    with m2:
        st.metric("Latest test Brier · rolling 30d ↓", pretty_metric(latest_roll))
        st.metric("Test positive rate", pretty_metric(report.metrics["positive_rate"], percent=True))

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
        name=f"ML · {report.model_name}", line=dict(color=current_source_color(), width=2.8),
        hovertemplate="%{x|%d %b %Y, %H:%M} Kyiv<br>ML: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["rolling_30d_probability"], mode="lines",
        name="Trailing 30 days", line=dict(color=chart_accent("rolling_30"), width=2, dash="dot"),
        hovertemplate="%{x|%d %b %Y, %H:%M} Kyiv<br>Trailing 30 days: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["rolling_7d_probability"], mode="lines",
        name="Trailing 7 days", line=dict(color=chart_accent("rolling_7"), width=1.6, dash="dash"),
        hovertemplate="%{x|%d %b %Y, %H:%M} Kyiv<br>Trailing 7 days: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["hour_of_day_probability"], mode="lines",
        name="Historical hour-of-day", line=dict(color=chart_accent("hour"), width=1.1, dash="dash"),
        opacity=0.8,
        hovertemplate="%{x|%d %b %Y, %H:%M} Kyiv<br>Hour-of-day: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Predicted probability", range=[0, 1], tickformat=".0%"),
        xaxis_title="Forecast origin · Kyiv time", legend=dict(orientation="h", y=-0.24),
    )
    st.plotly_chart(chart_defaults(fig, 380), use_container_width=True, config={"responsive": True})

    st.markdown("##### What actually happened in each forecast window")
    observed = go.Figure()
    observed.add_trace(go.Scatter(
        x=test["origin_kyiv"], y=test["actual"], mode="markers",
        name="Recorded alert start (1=yes)",
        marker=dict(
            color=[chart_accent("observed_yes") if x == 1 else chart_accent("observed_no") for x in test["actual"]],
            size=8, opacity=0.85,
        ),
        hovertemplate="%{x|%d %b %Y, %H:%M} Kyiv<br>At least one new recorded start: %{y}<extra></extra>",
    ))
    observed.update_layout(
        yaxis=dict(title="Observed label", range=[-0.22, 1.22], tickvals=[0, 1], ticktext=["No start", "Start"]),
        xaxis_title="Forecast origin · Kyiv time", showlegend=False,
    )
    st.plotly_chart(chart_defaults(observed, 210), use_container_width=True, config={"responsive": True})
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
            "ML": current_source_color(), "Past overall frequency": chart_accent("overall"),
            "Past hour-of-day frequency": chart_accent("hour"),
            "Trailing 7-day frequency": chart_accent("rolling_7"),
            "Trailing 30-day frequency": chart_accent("rolling_30"),
        },
    )
    fold_fig.update_layout(xaxis_title="Test interval", legend_title=None, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(chart_defaults(fold_fig, 400), use_container_width=True, config={"responsive": True})
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
                line=dict(dash="dash", color=chart_accent("reference")),
                hovertemplate="Perfect calibration<extra></extra>",
            ))
            cal.add_trace(go.Scatter(
                x=reliability["predicted"], y=reliability["observed"],
                mode="markers+lines", name="Held-out ML calibration",
                customdata=reliability["count"],
                marker=dict(size=10, color=current_source_color()),
                hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Samples %{customdata}<extra></extra>",
            ))
            cal.update_layout(
                xaxis=dict(title="Mean predicted probability", range=[0, 1]),
                yaxis=dict(title="Observed positive rate", range=[0, 1]),
            )
            st.plotly_chart(chart_defaults(cal, 310), use_container_width=True, config={"responsive": True})
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


# Render UI
render_global_header()
apply_layout_css()
st.warning(
    "**Research tool only.** Historical records and future experimental forecasts "
    "must never replace official air raid alerts or emergency guidance.", icon="⚠️"
)
page = sidebar_nav()
source = current_source()

if page == "Analytics":
    analytics_tab(source)
else:
    predictions_tab(source)

st.divider()
st.caption(
    "AirAlert Analytics · All plotted timestamps use Kyiv local time; preprocessing "
    "stores UTC timestamps. Volunteer and official sources are never silently mixed. "
    "Research only; follow official alerts for safety."
)
