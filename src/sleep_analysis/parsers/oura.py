"""Parser for Oura CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ensure_required_columns, finalize_dataframe


def parse_oura(path: str | Path) -> pd.DataFrame:
    """Parse an Oura sleep export into the unified schema."""
    path = Path(path)
    df = pd.read_csv(path)

    required = {
        "sleep_id",
        "user_id",
        "bedtime_start",
        "bedtime_end",
        "duration_min",
        "efficiency",
        "is_long_nap",
        "timezone",
    }
    ensure_required_columns(df, required, "oura")

    df = df.rename(
        columns={
            "sleep_id": "id",
            "bedtime_start": "start_ts",
            "bedtime_end": "end_ts",
            "is_long_nap": "is_nap",
            "timezone": "tz",
        }
    )

    date_cols = ["start_ts", "end_ts"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")

    df["is_nap"] = df["is_nap"].fillna(False)

    return finalize_dataframe(df, source="oura")
