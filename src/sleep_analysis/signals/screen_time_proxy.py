"""Compute screen-time proxy metrics from late activity logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def build_screen_time_proxy(path: str | Path, late_hour: int = 22) -> pd.DataFrame:
    """Return per-user daily proxies for evening screen usage.

    Accepts either event-level data with a `timestamp` column or pre-aggregated
    data with `date` and `late_minutes`. Late usage is defined by timestamps
    occurring on or after ``late_hour`` in local time.
    """

    frame = pd.read_csv(path)
    columns = set(frame.columns)

    if "timestamp" in columns:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["date"] = frame["timestamp"].dt.floor("D")
        frame["hour"] = frame["timestamp"].dt.hour
        grouped = (
            frame.groupby(["user_id", "date"])
            .agg(
                late_minutes=("timestamp", lambda s: (s.dt.hour >= late_hour).sum()),
                total_events=("timestamp", "count"),
            )
            .reset_index()
        )
        grouped["late_screen_flag"] = grouped["late_minutes"] > 0
        grouped["date"] = pd.to_datetime(grouped["date"]).dt.tz_localize("UTC")
        return grouped[["user_id", "date", "late_minutes", "late_screen_flag", "total_events"]]

    required: Iterable[str] = ("user_id", "date")
    missing = set(required) - columns
    if missing:
        raise ValueError(f"Screen-time proxy export missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")
    frame["late_minutes"] = pd.to_numeric(frame.get("late_minutes"), errors="coerce").fillna(0)
    frame["late_screen_flag"] = frame["late_minutes"] > 0
    if "total_events" not in frame.columns:
        frame["total_events"] = pd.to_numeric(frame.get("total_events"), errors="coerce").fillna(0)
    return frame[["user_id", "date", "late_minutes", "late_screen_flag", "total_events"]]
