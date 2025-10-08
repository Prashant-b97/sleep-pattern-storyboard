"""Prophet-style baseline with graceful fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import json

import numpy as np
import pandas as pd

from .artifacts import ModelArtifact

try:  # pragma: no cover - optional dependency
    from prophet import Prophet  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from fbprophet import Prophet  # type: ignore
    except ImportError:  # pragma: no cover
        Prophet = None  # type: ignore


def _prepare_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(df["date"], utc=True).dt.tz_convert(None),
            "y": df[target].astype(float),
        }
    ).dropna()
    return frame


def fit(
    df: pd.DataFrame,
    target: str,
    *,
    horizon: int = 7,
    output_dir: Path,
    backtest_window: Optional[int] = None,
    seasonality_mode: str = "multiplicative",
    growth: str = "linear",
    changepoint_prior_scale: float = 0.05,
) -> ModelArtifact:
    """Fit a Prophet model or fallback to a rolling mean baseline."""

    frame = _prepare_frame(df, target)
    if frame.empty:
        raise ValueError("No data available for Prophet baseline.")

    horizon = min(horizon, max(1, len(frame) // 3))
    if backtest_window is None:
        backtest_window = min(30, len(frame))

    train_end = max(len(frame) - horizon, len(frame) - backtest_window)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[train_end:]

    forecast_df = pd.DataFrame()
    model_path: Optional[str] = None

    if Prophet is not None and len(train_df) >= 10:
        model = Prophet(
            seasonality_mode=seasonality_mode,
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.fit(train_df)
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        forecast_df = forecast.tail(len(test_df))[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecast_df = forecast_df.rename(
            columns={"ds": "date", "yhat": "prediction", "yhat_lower": "lower", "yhat_upper": "upper"}
        )
        forecast_df["actual"] = test_df.set_index("ds")["y"].reindex(forecast_df["date"]).values

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir.parent / "model_artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / f"prophet_{target}.json")
        with open(model_path, "w") as fh:
            json.dump(model.params, fh, default=float)
    else:
        # Rolling mean fallback
        window = min(7, len(train_df))
        rolling_mean = train_df["y"].rolling(window=window, min_periods=1).mean().iloc[-1]
        idx = test_df["ds"] if len(test_df) else frame["ds"].tail(horizon)
        forecast_df = pd.DataFrame(
            {
                "date": idx,
                "prediction": rolling_mean,
            }
        )
        forecast_df["lower"] = rolling_mean - 0.5
        forecast_df["upper"] = rolling_mean + 0.5
        forecast_df["actual"] = test_df["y"].values if len(test_df) else np.nan

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    mae = float(np.abs(forecast_df["prediction"] - forecast_df["actual"]).dropna().mean())
    denom = forecast_df["actual"].replace(0, np.nan)
    mape = float(np.abs((forecast_df["prediction"] - forecast_df["actual"]) / denom).dropna().mean())

    coverage_series = (forecast_df["actual"] >= forecast_df["lower"]) & (
        forecast_df["actual"] <= forecast_df["upper"]
    )
    coverage = float(coverage_series.dropna().mean()) if not coverage_series.dropna().empty else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"prophet_{target}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    metrics = {"mae": mae, "mape": mape, "coverage": coverage}
    artifact = ModelArtifact(
        model_name="prophet",
        target=target,
        metrics=metrics,
        model_path=model_path,
        forecast_path=str(forecast_path),
        forecast=forecast_df,
    )
    return artifact
