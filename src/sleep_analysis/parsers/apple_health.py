"""Parser for Apple Health sleep exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ensure_required_columns, finalize_dataframe


def parse_apple_health(path: str | Path) -> pd.DataFrame:
    """Parse an Apple Health sleep export into the unified schema."""
    path = Path(path)
    df = pd.read_csv(path)

    required = {
        "uuid",
        "user_id",
        "start_date",
        "end_date",
        "duration_min",
        "efficiency",
        "nap",
        "timezone",
    }
    ensure_required_columns(df, required, "apple_health")

    df = df.rename(
        columns={
            "uuid": "id",
            "start_date": "start_ts",
            "end_date": "end_ts",
            "nap": "is_nap",
            "timezone": "tz",
        }
    )

    for col in ["start_ts", "end_ts"]:
        df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")

    df["is_nap"] = df["is_nap"].fillna(False)

    return finalize_dataframe(df, source="apple_health")
