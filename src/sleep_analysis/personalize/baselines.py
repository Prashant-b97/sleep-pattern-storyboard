"""Personalization utilities for user-specific baselines."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def compute_baselines(
    df: pd.DataFrame,
    *,
    user_column: str = "user_id",
    date_column: str = "date",
    value_columns: Sequence[str] = ("sleep_hours", "hrv_rmssd", "steps"),
    window: int = 30,
) -> pd.DataFrame:
    """Return dataframe with rolling baselines and z-scores per user."""

    frame = df.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.sort_values([user_column, date_column])

    min_periods = min(window, 7)

    for column in value_columns:
        if column not in frame.columns:
            continue
        baseline_col = f"{column}_baseline"
        std_col = f"{column}_baseline_std"
        z_col = f"{column}_zscore"

        grouped = frame.groupby(user_column)[column]
        frame[baseline_col] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean()
        )
        frame[std_col] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).std(ddof=0)
        )
        frame[z_col] = (frame[column] - frame[baseline_col]) / frame[std_col]
        frame[z_col] = frame[z_col].fillna(0.0)

    return frame
