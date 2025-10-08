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

st.set_page_config(page_title="Sleep Storyboard — What-if", page_icon="🤔", layout="wide")

st.sidebar.page_link("Home.py", label="🏠 Home")
st.sidebar.page_link("pages/Analyst.py", label="📊 Analyst")
st.sidebar.page_link("pages/Trends.py", label="📈 Trends")
st.sidebar.page_link("pages/Coach.py", label="🧭 Coach")
st.sidebar.page_link("pages/WhatIf.py", label="🤔 What-if")

settings = load_settings()
features_path = Path(settings.get("app", "features_path", default="data/processed/features"))

st.title("What-if Playground")

files = sorted(features_path.glob("*.parquet"))
if not files:
    st.warning("Run feature store build first to unlock simulations.")
    st.stop()

df = pd.read_parquet(files[-1])
average_sleep = df.get("sleep_hours", pd.Series([7.0])).mean()

st.write("Adjust evening habits to project their impact on tonight's sleep duration.")

bedtime_shift = st.slider("Bedtime shift (minutes)", min_value=-120, max_value=120, value=0, step=15)
caffeine_flag = st.checkbox("Consumed caffeine after 2pm", value=False)

predicted_delta = -0.5 * (bedtime_shift / 60)
if caffeine_flag:
    predicted_delta -= 0.3

projected_sleep = max(0, average_sleep + predicted_delta)

col1, col2 = st.columns(2)
col1.metric("Baseline sleep (hrs)", f"{average_sleep:0.2f}")
col2.metric("Projected tonight (hrs)", f"{projected_sleep:0.2f}", delta=f"{predicted_delta:0.2f}")

st.caption("Assumptions: every hour of later bedtime costs ~0.5h of sleep. Afternoon caffeine subtracts an additional 0.3h.")
