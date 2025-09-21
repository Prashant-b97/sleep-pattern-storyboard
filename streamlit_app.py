"""Story-driven Streamlit experience for the Sleep Analysis toolkit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st

from sleep_analysis.cli import load_primary, load_secondary

try:
    from statsmodels.tsa.ar_model import AutoReg
except ImportError:  # pragma: no cover
    AutoReg = None

st.set_page_config(
    page_title="Sleep Pattern Storyboard",
    page_icon="🌙",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_primary(path: Path) -> pd.DataFrame:
    return load_primary(path)


@st.cache_data(show_spinner=False)
def read_secondary(path: Path) -> pd.DataFrame:
    return load_secondary(path)


def parse_known_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--primary", default="data/raw/sleep_data.csv")
    parser.add_argument("--secondary", default="data/raw/sleep_data_new.csv")
    parser.add_argument("--lags", type=int, default=30)
    parser.add_argument("--forecast-horizon", type=int, default=14)
    args, _ = parser.parse_known_args()
    return args


def highlight_metrics(df: pd.DataFrame, mode: str) -> None:
    st.markdown("## 1. How are we sleeping?")

    total_nights = len(df)
    avg_sleep = df["sleep_hours"].mean()
    median_sleep = df["sleep_hours"].median()
    std_sleep = df["sleep_hours"].std()

    min_row = df.loc[df["sleep_hours"].idxmin()]
    max_row = df.loc[df["sleep_hours"].idxmax()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracked nights", total_nights, help="How many nights are in this export")
    col2.metric(
        "Average sleep",
        f"{avg_sleep:0.2f} h",
        help="Mean nightly sleep across the full timeline",
    )
    col3.metric(
        "Night-to-night swing",
        f"±{std_sleep:0.2f} h",
        help="Standard deviation reveals how inconsistent the pattern is",
    )

    col4, col5 = st.columns(2)
    col4.write(
        f"Shortest night: **{min_row.sleep_hours:0.2f} h** on {min_row.date.date()}"
    )
    col5.write(
        f"Longest night: **{max_row.sleep_hours:0.2f} h** on {max_row.date.date()}"
    )

    sub5 = (df["sleep_hours"] < 5).sum()
    long_nights = (df["sleep_hours"] > 9).sum()
    col6, col7 = st.columns(2)
    col6.write(f"Nights under 5 hours: **{sub5}** (when recovery is at risk)")
    col7.write(f"Nights over 9 hours: **{long_nights}** (possible catch-up sleep)")

    if mode == "Beginner walkthrough":
        with st.expander("What do these numbers mean?", expanded=True):
            st.markdown(
                "- **Average sleep** shows the typical night. Try nudging it toward the 7–9 hour range.\n"
                "- **Night-to-night swing** highlights volatility. A lower value means a steadier routine.\n"
                "- **Under 5 hours** can trigger fatigue the next day. Use it to spot weeks that need attention."
            )
    else:
        st.caption(
            "Std deviation computed on the cleaned series (NaNs removed). Extremes can indicate naps or device errors—consider filtering before modelling."
        )


def beginner_insights(df: pd.DataFrame) -> None:
    st.markdown("## 2. Beginner Playbook")
    healthy = df["sleep_hours"].between(7, 9)
    ratio = healthy.mean()
    st.success(
        f"{ratio*100:0.1f}% of nights hit the recommended 7–9 hour zone."
    )
    streaks = (healthy != healthy.shift()).cumsum()
    healthy_streaks = healthy.groupby(streaks).transform("size")[healthy]
    best_streak = int(healthy_streaks.max()) if not healthy_streaks.empty else 0
    st.write(
        f"Longest streak staying in the healthy range: **{best_streak} nights**."
    )
    st.markdown(
        "**Try this tonight:** jot down the bedtime and wake-up for your shortest nights and compare them with longer ones—small shifts often create big streaks."
    )


def expert_insights(df: pd.DataFrame) -> None:
    st.markdown("## 3. Analyst Notebook")
    with st.expander("Weekday pattern matrix"):
        weekday_pivot = (
            df.assign(weekday=lambda x: x["date"].dt.day_name())
            .groupby("weekday")["sleep_hours"]
            .agg(["count", "mean", "std"])
            .rename(columns={"mean": "avg_hours", "std": "std_hours"})
        )
        weekday_pivot = weekday_pivot.reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
        st.dataframe(weekday_pivot.style.format({"avg_hours": "{:.2f}", "std_hours": "{:.2f}"}))
    st.info(
        "Export the cleaned series with the button in the sidebar to replicate this analysis in pandas or SQL."
    )


def show_time_series(df: pd.DataFrame, mode: str) -> None:
    st.markdown("## 4. How the story unfolds over time")
    st.line_chart(df.set_index("date")["sleep_hours"], height=300)

    if mode == "Beginner walkthrough":
        st.caption(
            "Each spike represents a longer night. Hover to see exact dates; look for clusters to understand routines."
        )

    st.markdown("## 5. Distribution of sleep lengths")
    hist = (
        pd.cut(df["sleep_hours"], bins=20)
        .value_counts()
        .sort_index()
        .rename_axis("hours_bin")
        .reset_index(name="counts")
    )
    hist["midpoint"] = hist["hours_bin"].apply(lambda interval: interval.mid)
    st.bar_chart(hist.set_index("midpoint")["counts"], height=300)
    if mode == "Beginner walkthrough":
        st.caption(
            "Bars show how often you land in each duration. A tall bar between 7–8 hours means that's your sweet spot."
        )


def run_autoreg_forecast(series: pd.Series, lags: int, horizon: int) -> pd.DataFrame | None:
    if AutoReg is None or len(series) <= lags:
        return None
    series = series.asfreq("D")
    if series.isna().any():
        series = series.interpolate(limit_direction="both")
    model = AutoReg(series, lags=lags, old_names=False)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"date": forecast_index, "forecast_hours": forecast})


def render_forecast(df: pd.DataFrame, lags: int, horizon: int, mode: str) -> None:
    series = df.set_index("date")["sleep_hours"]
    forecast_df = run_autoreg_forecast(series, lags, horizon)
    if forecast_df is None:
        st.info("Install statsmodels to unlock the AutoReg forecast inside Streamlit.")
        return
    st.subheader("AutoReg Forecast")
    st.dataframe(forecast_df, width="stretch")
    st.line_chart(
        pd.concat(
            [
                series.rename("history"),
                forecast_df.set_index("date")["forecast_hours"].rename("forecast"),
            ]
        ),
        height=300,
    )
    first = forecast_df.iloc[0]
    st.success(
        f"Next-day forecast ({first.date.date()}): **{first.forecast_hours:0.2f} h**"
    )
    if mode == "Beginner walkthrough":
        st.caption(
            "Forecasts look forward assuming your recent pattern continues. Use it as a guide, not a guarantee."
        )
    else:
        st.caption(
            "AutoReg fit uses ordinary least squares with lag={lags}. Consider differencing if you observe strong trend or seasonality."
        )


def show_secondary(df: pd.DataFrame) -> None:
    st.subheader("Session-Level Distribution")
    st.line_chart(df.set_index("creationDate")["hours"], height=300)
    st.bar_chart(df["sourceName"].value_counts(), height=300)


def main() -> None:
    args = parse_known_args()
    st.title("Sleep Pattern Storyboard")
    st.caption("A guided tour from raw tracker data to actionable habits")

    st.sidebar.title("Control Room")
    st.sidebar.write("Tune the model and grab artefacts for deeper dives.")
    audience = st.sidebar.radio(
        "Audience mode",
        ("Beginner walkthrough", "Analyst deep dive"),
        help="Switch between plain-language guidance and technical detail",
    )
    lag_slider = st.sidebar.slider(
        "AutoReg lags",
        min_value=7,
        max_value=60,
        value=args.lags,
        step=1,
    )
    horizon_slider = st.sidebar.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=30,
        value=args.forecast_horizon,
        step=1,
    )

    primary_path = Path(args.primary)
    if not primary_path.exists():
        st.error(f"Primary dataset not found: {primary_path}")
        return

    primary_df = read_primary(primary_path)
    csv_bytes = primary_df.to_csv(index=False).encode()
    st.sidebar.download_button(
        "Download cleaned primary CSV",
        data=csv_bytes,
        file_name="sleep_primary_clean.csv",
        mime="text/csv",
    )

    highlight_metrics(primary_df, audience)
    if audience == "Beginner walkthrough":
        beginner_insights(primary_df)
    else:
        expert_insights(primary_df)

    show_time_series(primary_df, audience)
    render_forecast(primary_df, lag_slider, horizon_slider, audience)

    secondary_path = Path(args.secondary)
    if secondary_path.exists():
        secondary_df = read_secondary(secondary_path)
        if "hours" in secondary_df.columns:
            show_secondary(secondary_df)


if __name__ == "__main__":
    main()
