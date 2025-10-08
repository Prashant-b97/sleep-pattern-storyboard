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

st.set_page_config(page_title="Sleep Storyboard — Coach", page_icon="🧭", layout="wide")

st.sidebar.page_link("Home.py", label="🏠 Home")
st.sidebar.page_link("pages/Analyst.py", label="📊 Analyst")
st.sidebar.page_link("pages/Trends.py", label="📈 Trends")
st.sidebar.page_link("pages/Coach.py", label="🧭 Coach")
st.sidebar.page_link("pages/WhatIf.py", label="🤔 What-if")

settings = load_settings()
features_path = Path(settings.get("app", "features_path", default="data/processed/features"))

st.title("Coaching Playbook")

files = sorted(features_path.glob("*.parquet"))
if not files:
    st.warning("Build the health feature store first (CLI build-features command).")
    st.stop()

df = pd.read_parquet(files[-1])
df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)

avg_sleep = df.get("sleep_hours", pd.Series([7])).mean()
recent_debt = df.get("sleep_debt_roll7", pd.Series([0])).mean()
weekend_variance = (
    df.groupby(df["date"].dt.weekday >= 5)["sleep_hours"].mean()
    .diff()
    .abs()
    .sum()
    if "sleep_hours" in df.columns
    else 0
)

insights = []
if avg_sleep < 7:
    insights.append("Average sleep is below 7 hours. Schedule a 30-minute earlier wind-down routine on weekdays.")
else:
    insights.append("Average sleep meets the 7 hour mark—maintain your current bedtime routine.")

if recent_debt and recent_debt > 2:
    insights.append("Seven-day sleep debt is trending high. Block a mid-week recovery night with reduced screen time.")
else:
    insights.append("Sleep debt is under control. Keep consistent bedtimes to protect this margin.")

if weekend_variance > 1.5:
    insights.append("Weekend sleep differs by more than 1.5 hours. Align Saturday wake times closer to weekdays to reduce social jetlag.")
else:
    insights.append("Weekend schedule stays close to weekdays—jetlag risk is low.")

st.markdown("### Top actions for this week")
for idx, tip in enumerate(insights[:3], start=1):
    st.write(f"{idx}. {tip}")
