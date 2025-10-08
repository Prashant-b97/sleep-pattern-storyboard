"""Fuse curated sleep data with health signals to build feature store tables."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..signals import (
    build_screen_time_proxy,
    load_hrv_signals,
    load_step_signals,
    load_tags,
)
from ..personalize.baselines import compute_baselines


def _load_curated_sleep(parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("**/*.parquet"))
    if not files:
        raise ValueError(f"No curated parquet files found under {parquet_dir}")
    frames = [pd.read_parquet(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    if "start_ts" not in df.columns:
        raise ValueError("Curated dataset must expose start_ts column.")
    df["date"] = pd.to_datetime(df.get("date")) if "date" in df.columns else df["start_ts"].dt.floor("D")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["user_id"] = df["user_id"].astype(str)
    return df


def build_health_feature_store(
    parquet_dir: Path,
    *,
    hrv_path: Optional[Path] = None,
    steps_path: Optional[Path] = None,
    tags_path: Optional[Path] = None,
    screen_time_path: Optional[Path] = None,
    output_dir: Path,
) -> Path:
    """Join sleep dataset with optional health signals and persist parquet output."""

    sleep_df = _load_curated_sleep(parquet_dir)
    base = sleep_df[["user_id", "date", "duration_min", "efficiency", "is_weekend"]].copy()
    base["sleep_hours"] = base["duration_min"] / 60.0

    merged = base

    if hrv_path and Path(hrv_path).exists():
        hrv_df = load_hrv_signals(hrv_path)
        merged = merged.merge(hrv_df, on=["user_id", "date"], how="left")

    if steps_path and Path(steps_path).exists():
        steps_df = load_step_signals(steps_path)
        merged = merged.merge(steps_df, on=["user_id", "date"], how="left")

    if tags_path and Path(tags_path).exists():
        tags_df = load_tags(tags_path)
        merged = merged.merge(tags_df, on=["user_id", "date"], how="left")

    if screen_time_path and Path(screen_time_path).exists():
        screen_df = build_screen_time_proxy(screen_time_path)
        merged = merged.merge(screen_df, on=["user_id", "date"], how="left")

    merged = merged.sort_values(["user_id", "date"]).reset_index(drop=True)
    merged = compute_baselines(
        merged,
        value_columns=[
            "sleep_hours",
            "hrv_rmssd",
            "hrv_sdnn",
            "steps",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    feature_path = output_dir / f"health_features_{run_ts}.parquet"
    merged.to_parquet(feature_path, index=False)

    schema_doc = {
        "generated_at": run_ts,
        "rows": len(merged),
        "columns": merged.columns.tolist(),
        "parquet_path": str(feature_path),
    }
    (output_dir / "SCHEMA.json").write_text(json.dumps(schema_doc, indent=2))

    return feature_path
