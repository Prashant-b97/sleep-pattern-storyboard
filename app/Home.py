from __future__ import annotations

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT.parent / "src"
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from sleep_analysis.config import load_settings

from components import distribution_chart, parquet_selector, sleep_trend_chart  # noqa: E402

st.set_page_config(page_title="Sleep Storyboard — Home", page_icon="🌙", layout="wide")

st.sidebar.page_link("Home.py", label="🏠 Home")
st.sidebar.page_link("pages/Analyst.py", label="📊 Analyst")
st.sidebar.page_link("pages/Trends.py", label="📈 Trends")
st.sidebar.page_link("pages/Coach.py", label="🧭 Coach")
st.sidebar.page_link("pages/WhatIf.py", label="🤔 What-if")

settings = load_settings()
parquet_dir = Path(settings.get("app", "default_parquet_dir", default="data/processed/parquet"))

st.title("Sleep Pattern Storyboard")
st.caption("Monitor nightly rhythms and surface insights across curated pipelines.")

selected = parquet_selector("Select a curated parquet run", parquet_dir)
if not selected:
    st.stop()

df = pd.read_parquet(selected)
df["date"] = pd.to_datetime(df["start_ts"]).dt.tz_convert(None)

col1, col2, col3 = st.columns(3)
col1.metric("Tracked nights", len(df))
col2.metric("Average sleep (hrs)", f"{(df['duration_min'] / 60).mean():0.2f}")
col3.metric("Median efficiency", f"{df['efficiency'].median():0.2f}")

st.subheader("Timeline snapshot")
df_summary = df.rename(columns={"duration_min": "sleep_hours"}).copy()
df_summary["sleep_hours"] = df_summary["sleep_hours"] / 60
sleep_trend_chart(df_summary[["date", "sleep_hours", "is_weekend"]], highlight_weekends=True)

st.subheader("Distribution of sleep lengths")
distribution_chart(df_summary)

st.info(
    "Navigate via the sidebar to deep dive into model metrics, trends, coaching insights, or run what-if scenarios."
)
