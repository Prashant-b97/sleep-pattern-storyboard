"""Manual context tags for sleep analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS: Iterable[str] = ("user_id", "date")
TAG_COLUMNS: Iterable[str] = ("caffeine", "alcohol", "late_meal", "work_travel")


def load_tags(path: str | Path) -> pd.DataFrame:
    """Load manual CSV tags and coerce to boolean indicators."""

    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Tags file missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize("UTC")

    for column in TAG_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool)
        else:
            frame[column] = False

    columns = ["user_id", "date", *TAG_COLUMNS]
    return frame[columns]
