"""Temporal Fusion Transformer placeholder implementation.

The full TFT model is heavy and requires dedicated deep learning frameworks.
This module provides a lightweight scaffold that produces deterministic
forecasts while signalling where a production-ready model would slot in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import ModelArtifact


def fit(
    df: pd.DataFrame,
    target: str,
    *,
    horizon: int = 7,
    output_dir: Path,
    feature_columns: Optional[Sequence[str]] = None,
    notes: str = "TFT scaffold - replace with full implementation when ready.",
) -> ModelArtifact:
    """Return a deterministic rolling-average forecast with TODO markers."""

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in dataframe.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")

    if feature_columns is None:
        feature_columns = [
            col
            for col in df.columns
            if col not in {"date", target, "user_id", "start_ts", "end_ts", "source"}
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    horizon = min(horizon, max(1, len(df) // 3))
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    rolling_window = min(7, len(train))
    rolling_mean = train[target].rolling(window=rolling_window, min_periods=1).mean()
    last_value = rolling_mean.iloc[-1] if not rolling_mean.empty else float(train[target].iloc[-1])
    predictions = np.full(len(test), last_value)

    forecast_df = pd.DataFrame(
        {
            "date": test["date"],
            "prediction": predictions,
            "actual": test[target].values,
        }
    )

    error = (forecast_df["prediction"] - forecast_df["actual"]).abs()
    mae = float(error.mean())
    denom = forecast_df["actual"].replace(0, np.nan)
    mape_series = np.abs((forecast_df["prediction"] - forecast_df["actual"]) / denom)
    mape = float(mape_series.dropna().mean()) if not mape_series.dropna().empty else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"tft_{target}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    artifact = ModelArtifact(
        model_name="tft",
        target=target,
        metrics={"mae": mae, "mape": mape, "coverage": float("nan")},
        model_path=None,
        forecast_path=str(forecast_path),
        forecast=forecast_df,
        diagnostics={
            "status": "placeholder",
            "notes": notes,
            "feature_columns": list(feature_columns),
        },
    )
    return artifact
