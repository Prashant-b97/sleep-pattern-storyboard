from __future__ import annotations

from pathlib import Path

import pandas as pd

from sleep_analysis.features.health_store import build_health_feature_store
from sleep_analysis.personalize.baselines import compute_baselines


def _write_parquet(tmp_path: Path) -> Path:
    sleep_dir = tmp_path / "parquet"
    sleep_dir.mkdir()
    df = pd.DataFrame(
        {
            "user_id": ["user123", "user123", "user123"],
            "start_ts": pd.to_datetime(
                ["2024-01-01T22:30:00Z", "2024-01-02T22:45:00Z", "2024-01-03T23:15:00Z"]
            ),
            "end_ts": pd.to_datetime(
                ["2024-01-02T06:20:00Z", "2024-01-03T06:30:00Z", "2024-01-04T06:40:00Z"]
            ),
            "duration_min": [470, 455, 440],
            "efficiency": [0.82, 0.84, 0.79],
            "is_weekend": [False, False, False],
        }
    )
    df.to_parquet(sleep_dir / "part.parquet", index=False)
    return sleep_dir


def _write_signals(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "signals"
    base.mkdir()

    hrv = pd.DataFrame(
        {
            "user_id": ["user123"] * 3,
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "rmssd": [75, 72, 70],
            "sdnn": [95, 92, 90],
            "baseline_rmssd": [70, 70, 70],
        }
    )
    hrv_path = base / "hrv.csv"
    hrv.to_csv(hrv_path, index=False)

    steps = pd.DataFrame(
        {
            "user_id": ["user123"] * 3,
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "steps": [9000, 8000, 7500],
            "active_minutes": [60, 55, 50],
        }
    )
    steps_path = base / "steps.csv"
    steps.to_csv(steps_path, index=False)

    tags = pd.DataFrame(
        {
            "user_id": ["user123"] * 3,
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "caffeine": [True, False, False],
        }
    )
    tags_path = base / "tags.csv"
    tags.to_csv(tags_path, index=False)

    screen = pd.DataFrame(
        {
            "user_id": ["user123", "user123"],
            "timestamp": ["2024-01-02T23:30:00", "2024-01-03T21:00:00"],
        }
    )
    screen_path = base / "screen.csv"
    screen.to_csv(screen_path, index=False)

    return {
        "hrv": hrv_path,
        "steps": steps_path,
        "tags": tags_path,
        "screen": screen_path,
    }


def test_health_feature_store_with_baselines(tmp_path: Path) -> None:
    parquet_dir = _write_parquet(tmp_path)
    signals = _write_signals(tmp_path)

    output_dir = tmp_path / "features_out"
    feature_path = build_health_feature_store(
        parquet_dir,
        hrv_path=signals["hrv"],
        steps_path=signals["steps"],
        tags_path=signals["tags"],
        screen_time_path=signals["screen"],
        output_dir=output_dir,
    )

    assert feature_path.exists()
    frame = pd.read_parquet(feature_path)
    expected_cols = {
        "sleep_hours",
        "hrv_rmssd",
        "steps",
        "caffeine",
        "late_screen_flag",
        "sleep_hours_baseline",
        "sleep_hours_zscore",
    }
    assert expected_cols.issubset(frame.columns)
    assert len(frame) == 3


def test_compute_baselines_handles_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "user_id": ["u"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "sleep_hours": [7.1, 7.0, 6.5, 8.0, 7.6],
        }
    )
    result = compute_baselines(df, value_columns=["sleep_hours"], window=3)
    assert "sleep_hours_baseline" in result.columns
    assert "sleep_hours_zscore" in result.columns
