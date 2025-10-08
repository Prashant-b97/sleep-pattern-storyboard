"""Daily step signal ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: Iterable[str] = ("user_id", "date", "steps")


def load_step_signals(path: str | Path) -> pd.DataFrame:
    """Return dataframe with per-user daily step counts."""

    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Steps export missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")
    frame["steps"] = pd.to_numeric(frame["steps"], errors="coerce").fillna(0).astype(int)
    frame["active_minutes"] = pd.to_numeric(frame.get("active_minutes"), errors="coerce")

    columns = ["user_id", "date", "steps", "active_minutes"]
    return frame[columns]
