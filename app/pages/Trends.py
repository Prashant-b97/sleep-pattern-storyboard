from __future__ import annotations

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = APP_ROOT.parent / "src"
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from sleep_analysis.config import load_settings
from sleep_analysis.detect import detect_change_points

from components import efficiency_chart, parquet_selector, sleep_trend_chart  # noqa: E402

st.set_page_config(page_title="Sleep Storyboard — Trends", page_icon="📈", layout="wide")

st.sidebar.page_link("Home.py", label="🏠 Home")
st.sidebar.page_link("pages/Analyst.py", label="📊 Analyst")
st.sidebar.page_link("pages/Trends.py", label="📈 Trends")
st.sidebar.page_link("pages/Coach.py", label="🧭 Coach")
st.sidebar.page_link("pages/WhatIf.py", label="🤔 What-if")

settings = load_settings()
parquet_dir = Path(settings.get("app", "default_parquet_dir", default="data/processed/parquet"))

st.title("Trend Explorer")

selected = parquet_selector("Select dataset", parquet_dir)
if not selected:
    st.stop()

df = pd.read_parquet(selected)
df["date"] = pd.to_datetime(df["start_ts"]).dt.tz_convert(None)
df["sleep_hours"] = df["duration_min"] / 60

st.subheader("Sleep duration over time")
sorted_df = df.sort_values("date")
change_indices = detect_change_points(sorted_df["sleep_hours"])
change_points = [sorted_df.iloc[idx]["date"] for idx in change_indices] if change_indices else []
sleep_trend_chart(df[["date", "sleep_hours", "is_weekend"]], change_points=change_points)

st.subheader("Sleep efficiency")
efficiency_chart(df[["date", "efficiency"]])

if change_points:
    readable = ", ".join(ts.strftime("%Y-%m-%d") for ts in change_points)
    st.caption(f"Regime shifts detected near: {readable}")
