"""Gradient boosted machine-style model leveraging circadian features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import ModelArtifact

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

try:  # pragma: no cover
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_absolute_percentage_error = None  # type: ignore


def _resolve_model(
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
) -> Tuple[object, str]:
    if lgb is not None:
        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
        )
        return model, "lightgbm"
    if XGBRegressor is not None:
        model = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            objective="reg:squarederror",
        )
        return model, "xgboost"
    if GradientBoostingRegressor is not None:
        model = GradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth if max_depth > 0 else 3,
            n_estimators=n_estimators,
        )
        return model, "sklearn_gbm"
    raise RuntimeError("No gradient boosting backend available.")


def fit(
    df: pd.DataFrame,
    target: str,
    *,
    horizon: int = 7,
    output_dir: Path,
    backtest_window: Optional[int] = None,
    feature_columns: Optional[Sequence[str]] = None,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    n_estimators: int = 100,
) -> ModelArtifact:
    """Train a gradient boosted regressor with expanding-window backtest."""

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column.")

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

    feature_frame = df.dropna(subset=feature_columns + [target]).reset_index(drop=True)
    if feature_frame.empty:
        raise ValueError("No rows available after dropping NA feature rows.")

    series = feature_frame[target].astype(float)
    horizon = min(horizon, max(1, len(series) // 3))
    if backtest_window is None:
        backtest_window = min(30, len(series))

    train_end = max(len(series) - horizon, len(series) - backtest_window)
    train_x = feature_frame.loc[: train_end - 1, feature_columns]
    train_y = series.iloc[:train_end]
    test_x = feature_frame.loc[train_end:, feature_columns]
    test_y = series.iloc[train_end:]

    model, backend = _resolve_model(learning_rate, max_depth, n_estimators)
    model.fit(train_x, train_y)
    preds = pd.Series(model.predict(test_x), index=test_y.index)

    mae = float(np.mean(np.abs(test_y - preds))) if len(test_y) else float("nan")
    if mean_absolute_percentage_error is not None and len(test_y):
        mape = float(mean_absolute_percentage_error(test_y, preds))
    else:
        denom = test_y.replace(0, np.nan)
        mape_series = np.abs((test_y - preds) / denom)
        mape = float(mape_series.dropna().mean()) if not mape_series.dropna().empty else float("nan")

    coverage = float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"gbm_{target}_forecast.csv"
    forecast_df = pd.DataFrame(
        {
            "date": feature_frame.loc[test_y.index, "date"],
            "prediction": preds.values,
            "actual": test_y.values,
        }
    )
    forecast_df.to_csv(forecast_path, index=False)

    model_dir = output_dir.parent / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"gbm_{target}_{backend}.json"
    try:
        # LightGBM/XGBoost/Sklearn support JSON dump or get_params
        params = getattr(model, "get_params", lambda: {})()
        with open(model_path, "w") as fh:
            json.dump(params, fh, default=float)
    except Exception:  # pragma: no cover
        model_path = None

    artifact = ModelArtifact(
        model_name="gbm",
        target=target,
        metrics={"mae": mae, "mape": mape, "coverage": coverage},
        model_path=str(model_path) if model_path else None,
        forecast_path=str(forecast_path),
        forecast=forecast_df,
    )
    artifact.diagnostics = {"backend": backend, "feature_columns": list(feature_columns)}
    return artifact
