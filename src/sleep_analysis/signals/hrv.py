"""Utilities to ingest heart-rate variability exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: Iterable[str] = ("user_id", "date", "rmssd", "sdnn")


def load_hrv_signals(path: str | Path) -> pd.DataFrame:
    """Load HRV exports and return tidy daily metrics."""

    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"HRV export missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")
    frame["rmssd"] = pd.to_numeric(frame["rmssd"], errors="coerce")
    frame["sdnn"] = pd.to_numeric(frame["sdnn"], errors="coerce")
    frame["baseline_rmssd"] = pd.to_numeric(frame.get("baseline_rmssd"), errors="coerce")
    frame = frame.dropna(subset=["rmssd"])

    frame = frame.rename(columns={"rmssd": "hrv_rmssd", "sdnn": "hrv_sdnn"})
    columns = ["user_id", "date", "hrv_rmssd", "hrv_sdnn", "baseline_rmssd"]
    return frame[columns]
