from __future__ import annotations

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = APP_ROOT.parent / "src"
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from sleep_analysis.config import load_settings
from sleep_analysis.reporting.pdf import generate_weekly_pdf

from components import export_dataframe, model_metrics_table  # noqa: E402

st.set_page_config(page_title="Sleep Storyboard — Analyst", page_icon="📊", layout="wide")

st.sidebar.page_link("Home.py", label="🏠 Home")
st.sidebar.page_link("pages/Analyst.py", label="📊 Analyst")
st.sidebar.page_link("pages/Trends.py", label="📈 Trends")
st.sidebar.page_link("pages/Coach.py", label="🧭 Coach")
st.sidebar.page_link("pages/WhatIf.py", label="🤔 What-if")

settings = load_settings()
reports_dir = Path(settings.get("app", "model_reports_dir", default="analysis_output/model_reports"))
exports_dir = Path(settings.exports_output_dir)

st.title("Analyst Workbench")

frame = model_metrics_table(reports_dir)

if not frame.empty:
    st.subheader("Export current view")
    col_csv, col_parquet = st.columns(2)
    with col_csv:
        if st.button("Export CSV", key="export_csv"):
            export_dataframe(frame, output_dir=exports_dir, basename="model_metrics", formats=["csv"])
    with col_parquet:
        if st.button("Export Parquet", key="export_parquet"):
            export_dataframe(frame, output_dir=exports_dir, basename="model_metrics", formats=["parquet"])

st.subheader("Weekly PDF report")
week_input = st.text_input("ISO week (optional)", placeholder="e.g. 2025-W40")
if st.button("Export weekly PDF report"):
    pdf_path = generate_weekly_pdf(reports_dir, settings.reports_output_dir, week=week_input or None)
    st.success(f"Weekly report generated at {pdf_path}")
