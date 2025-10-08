"""Utility helpers for vendor parsers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

COMMON_COLUMNS = [
    "id",
    "user_id",
    "start_ts",
    "end_ts",
    "duration_min",
    "efficiency",
    "is_nap",
    "tz",
    "source",
]


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    """Raise if the expected columns are missing so ingestion fails fast."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{source} export missing required columns: {sorted(missing)}")


def finalize_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return dataframe aligned to the unified schema."""
    df = df.copy()
    df["source"] = source
    df["is_nap"] = df["is_nap"].fillna(False).astype(bool)
    df["tz"] = df["tz"].fillna("UTC")
    df["efficiency"] = df["efficiency"].astype(float)
    df["duration_min"] = df["duration_min"].astype(float)
    df["id"] = df["id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df = df[COMMON_COLUMNS]
    return df
