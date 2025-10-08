from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sleep_analysis.features import build_circadian_features
from sleep_analysis.models import prophet, sarimax


def _synthetic_sleep(days: int = 60) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01", tz="America/Los_Angeles")
    records = []
    for idx in range(days):
        start = base + pd.Timedelta(days=idx, hours=22)
        duration = 420 + 10 * np.sin(idx / 6)
        end = start + pd.Timedelta(minutes=duration)
        records.append(
            {
                "id": f"rec_{idx}",
                "user_id": "user123",
                "start_ts": start,
                "end_ts": end,
                "duration_min": duration,
                "efficiency": 0.82 + 0.05 * np.cos(idx / 5),
                "is_nap": False,
                "tz": "America/Los_Angeles",
                "source": "synthetic",
            }
        )
    return pd.DataFrame(records)


def test_circadian_features_columns() -> None:
    raw = _synthetic_sleep()
    features = build_circadian_features(raw)
    required = {
        "sleep_hours",
        "sleep_hours_roll7",
        "sleep_hours_roll14",
        "social_jetlag",
        "day_of_week",
    }
    assert required.issubset(features.columns)


def test_sarimax_and_prophet_baselines(tmp_path: Path) -> None:
    raw = _synthetic_sleep()
    features = build_circadian_features(raw)
    features["date"] = pd.to_datetime(features["date"])

    sarimax_artifact = sarimax.fit(
        features,
        target="sleep_hours",
        horizon=7,
        backtest_window=21,
        output_dir=tmp_path,
    )
    assert sarimax_artifact.metrics["mae"] >= 0
    assert Path(sarimax_artifact.forecast_path).exists()

    prophet_artifact = prophet.fit(
        features,
        target="sleep_hours",
        horizon=7,
        backtest_window=21,
        output_dir=tmp_path,
    )
    assert prophet_artifact.metrics["mae"] >= 0
    assert Path(prophet_artifact.forecast_path).exists()
