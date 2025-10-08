from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import streamlit as st


def sleep_trend_chart(df: pd.DataFrame, *, highlight_weekends: bool = True, change_points: Optional[Iterable[pd.Timestamp]] = None) -> None:
    df = df.copy()
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    st.line_chart(df["sleep_hours"], height=280)

    if highlight_weekends and "is_weekend" in df.columns:
        weekend_ratio = df["is_weekend"].mean() * 100
        st.caption(f"Weekends make up {weekend_ratio:0.1f}% of records above.")

    if change_points:
        change_dates = ", ".join(ts.strftime("%Y-%m-%d") for ts in change_points)
        st.info(f"Regime changes detected around: {change_dates}")


def efficiency_chart(df: pd.DataFrame) -> None:
    if "efficiency" not in df.columns:
        st.warning("Efficiency column unavailable for this dataset.")
        return
    series = df.sort_values("date").set_index("date")["efficiency"]
    st.line_chart(series, height=220)


def distribution_chart(df: pd.DataFrame) -> None:
    hist = (
        pd.cut(df["sleep_hours"], bins=20)
        .value_counts()
        .sort_index()
        .rename_axis("hours_bin")
        .reset_index(name="counts")
    )
    hist["midpoint"] = hist["hours_bin"].apply(lambda interval: interval.mid)
    st.bar_chart(hist.set_index("midpoint")[["counts"]])
