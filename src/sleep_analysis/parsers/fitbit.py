"""Parser for Fitbit sleep exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ensure_required_columns, finalize_dataframe


def parse_fitbit(path: str | Path) -> pd.DataFrame:
    """Parse a Fitbit sleep export into the unified schema."""
    path = Path(path)
    df = pd.read_csv(path)

    required = {
        "log_id",
        "user_id",
        "start_time",
        "end_time",
        "minutes_asleep",
        "efficiency",
        "is_nap",
        "timezone",
    }
    ensure_required_columns(df, required, "fitbit")

    df = df.rename(
        columns={
            "log_id": "id",
            "start_time": "start_ts",
            "end_time": "end_ts",
            "minutes_asleep": "duration_min",
            "timezone": "tz",
        }
    )

    for col in ["start_ts", "end_ts"]:
        df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")

    df["is_nap"] = df["is_nap"].fillna(False)

    return finalize_dataframe(df, source="fitbit")
