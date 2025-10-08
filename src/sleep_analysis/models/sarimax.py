"""SARIMAX baseline model implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence
import warnings
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .artifacts import ModelArtifact


def _select_exog_columns(df: pd.DataFrame, target: str, exclude: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def fit(
    df: pd.DataFrame,
    target: str,
    *,
    horizon: int = 7,
    output_dir: Path,
    backtest_window: Optional[int] = None,
    exog_features: Optional[Sequence[str]] = None,
    order: Sequence[int] = (1, 0, 0),
    seasonal_order: Sequence[int] = (0, 1, 1, 7),
    trend: Optional[str] = "c",
    alpha: float = 0.05,
) -> ModelArtifact:
    """Fit a SARIMAX model and persist forecast artefacts."""

    if "date" not in df.columns:
        raise ValueError("DataFrame must include a 'date' column.")

    series = pd.Series(df[target].values, index=pd.to_datetime(df["date"], utc=True))
    series = series.sort_index()
    series = series.groupby(series.index).mean()
    series = series.asfreq("D")
    series = series.fillna(method="ffill")

    if exog_features is None:
        exog_features = _select_exog_columns(
            df, target, exclude=["date", "source", "user_id", "start_ts", "end_ts"]
        )

    feature_frame = df.set_index(pd.to_datetime(df["date"], utc=True))
    feature_frame = feature_frame.groupby(level=0).mean(numeric_only=True)
    feature_frame = feature_frame.reindex(series.index).fillna(method="ffill")
    if exog_features:
        available_features = [col for col in exog_features if col in feature_frame.columns]
        feature_frame = feature_frame[available_features] if available_features else None
    else:
        feature_frame = None

    horizon = min(horizon, max(1, len(series) // 3))
    if horizon <= 0:
        horizon = 1

    if backtest_window is None:
        backtest_window = min(30, len(series))

    train_end = max(len(series) - horizon, max(10, len(series) - backtest_window))
    train_series = series.iloc[:train_end]
    test_series = series.iloc[train_end:]

    if feature_frame is not None:
        train_exog = feature_frame.iloc[:train_end]
        test_exog = feature_frame.iloc[train_end:]
    else:
        train_exog = None
        test_exog = None

    # Fallback to naive mean if insufficient data for SARIMAX
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="Non-invertible")
            model = SARIMAX(
                train_series,
                exog=train_exog,
                order=tuple(order),
                seasonal_order=tuple(seasonal_order),
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            forecast_res = result.get_forecast(steps=len(test_series), exog=test_exog)
            predictions = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=alpha)
    except Exception:  # pragma: no cover - fallback path
        mean_value = float(train_series.mean())
        index = test_series.index if len(test_series) else series.index[-horizon:]
        predictions = pd.Series(mean_value, index=index)
        conf_int = pd.DataFrame(
            {
                "lower": predictions - 0.5,
                "upper": predictions + 0.5,
            },
            index=index,
        )
        result = None

    forecast_df = pd.DataFrame(
        {
            "date": predictions.index,
            "prediction": predictions.values,
        }
    )
    if len(test_series):
        forecast_df["actual"] = test_series.values
    else:
        forecast_df["actual"] = np.nan

    mae = float(np.mean(np.abs(forecast_df["actual"] - forecast_df["prediction"]).dropna()))
    denom = forecast_df["actual"].replace(0, np.nan)
    mape_series = np.abs((forecast_df["actual"] - forecast_df["prediction"]) / denom)
    mape = float(mape_series.dropna().mean()) if not mape_series.dropna().empty else float("nan")

    lower = conf_int.iloc[:, 0] if conf_int is not None else None
    upper = conf_int.iloc[:, 1] if conf_int is not None else None
    if lower is not None and upper is not None and len(test_series):
        within = (test_series >= lower) & (test_series <= upper)
        coverage = float(within.mean())
    else:
        coverage = float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir.parent / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path: Optional[str]
    if result is not None:
        model_path = str(model_dir / f"sarimax_{target}.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(result, fh)
    else:
        model_path = None

    forecast_path = output_dir / f"sarimax_{target}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    metrics = {"mae": mae, "mape": mape, "coverage": coverage}
    artifact = ModelArtifact(
        model_name="sarimax",
        target=target,
        metrics=metrics,
        model_path=model_path,
        forecast_path=str(forecast_path),
        forecast=forecast_df,
    )
    return artifact
