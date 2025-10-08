"""Circadian and rolling feature engineering utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TARGET_SLEEP_HOURS = 8.0


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_datetime(df[column], utc=True, errors="coerce")
    return series


def _groupwise_rolling(
    df: pd.DataFrame,
    column: str,
    group_cols: Sequence[str],
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    return (
        df.groupby(list(group_cols))[column]
        .transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean())
        .astype(float)
    )


def build_circadian_features(
    df: pd.DataFrame,
    *,
    user_column: str = "user_id",
    start_col: str = "start_ts",
    end_col: str = "end_ts",
    duration_col: str = "duration_min",
    efficiency_col: str = "efficiency",
) -> pd.DataFrame:
    """Return dataframe enriched with circadian and rolling statistics.

    The function expects timezone-aware timestamps. If naive timestamps are provided,
    they will be assumed to be UTC.
    """

    if user_column not in df.columns:
        df = df.copy()
        df[user_column] = "anonymous"
    else:
        df = df.copy()

    df[start_col] = _ensure_datetime(df, start_col)
    df[end_col] = _ensure_datetime(df, end_col)
    df = df.sort_values([user_column, start_col]).reset_index(drop=True)

    df["date"] = df[start_col].dt.floor("D")
    df["day_of_week"] = df[start_col].dt.dayofweek
    df["month"] = df[start_col].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5

    df["sleep_hours"] = df[duration_col] / 60.0
    df["sleep_deficit"] = np.maximum(0.0, TARGET_SLEEP_HOURS - df["sleep_hours"])

    for window in (7, 14):
        df[f"sleep_hours_roll{window}"] = _groupwise_rolling(
            df, "sleep_hours", [user_column], window=window
        )
        df[f"efficiency_roll{window}"] = _groupwise_rolling(
            df, efficiency_col, [user_column], window=window
        )
        df[f"sleep_debt_roll{window}"] = (
            df.groupby(user_column)["sleep_deficit"]
            .transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).sum()
            )
            .astype(float)
        )

    midpoint = df[start_col] + (pd.to_timedelta(df[duration_col], unit="m") / 2)
    df["midpoint_hour_local"] = midpoint.dt.tz_convert(None).dt.hour + midpoint.dt.tz_convert(
        None
    ).dt.minute / 60.0
    df["social_jetlag"] = (
        df.groupby(user_column)["midpoint_hour_local"]
        .transform(lambda s: (s - s.rolling(window=7, min_periods=1).mean()).abs())
        .fillna(0.0)
    )

    df = df.drop(columns=["sleep_deficit"]).copy()
    return df
