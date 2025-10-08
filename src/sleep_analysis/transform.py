"""Transformation utilities for unified sleep records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .schema import SleepRecord, schema_hash

DURATION_MIN = 120
DURATION_MAX = 720


@dataclass
class TransformResult:
    dataframe: pd.DataFrame
    out_path: Path
    version_pointer: Path


def _localize_series(series: pd.Series, tz_series: pd.Series) -> pd.Series:
    """Ensure timestamps are timezone-aware using the provided tz column."""
    localized = []
    for ts, tz_name in zip(series, tz_series):
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            localized.append(pd.NaT)
            continue

        timestamp = pd.to_datetime(ts, utc=False, errors="coerce")
        if pd.isna(timestamp):
            localized.append(pd.NaT)
            continue

        tz_key = str(tz_name) if pd.notna(tz_name) else "UTC"
        try:
            if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
                timestamp = timestamp.tz_localize(tz_key, nonexistent="shift_forward", ambiguous="NaT")
            else:
                timestamp = timestamp.tz_convert(tz_key)
            timestamp = timestamp.tz_convert("UTC")
            localized.append(timestamp)
        except (TypeError, AttributeError, ValueError):
            localized.append(pd.NaT)
    return pd.Series(localized, index=series.index)


def normalize(
    df: pd.DataFrame,
    duration_bounds: tuple[int, int] = (DURATION_MIN, DURATION_MAX),
) -> pd.DataFrame:
    """Normalize timestamps, clip durations, and derive helper flags."""
    df = df.copy()
    df["tz"] = df["tz"].fillna("UTC")
    df["start_ts"] = _localize_series(df["start_ts"], df["tz"])
    df["end_ts"] = _localize_series(df["end_ts"], df["tz"])

    lower, upper = duration_bounds
    clipped = df["duration_min"].clip(lower, upper)
    df["duration_was_clipped"] = clipped != df["duration_min"]
    df["duration_min"] = clipped

    df["is_weekend"] = df["start_ts"].dt.weekday >= 5
    df["is_nap"] = df["is_nap"].fillna(False).astype(bool)

    duration_std = df["duration_min"].std(ddof=0)
    if duration_std and duration_std > 0:
        df["duration_zscore"] = (df["duration_min"] - df["duration_min"].mean()) / duration_std
    else:
        df["duration_zscore"] = 0.0
    df["is_outlier_duration"] = df["duration_zscore"].abs() >= 3

    df = df.dropna(subset=["start_ts", "end_ts"])
    return df


def validate_records(df: pd.DataFrame) -> Iterable[SleepRecord]:
    """Yield validated SleepRecord instances for each row."""
    records = []
    for payload in df[SleepRecord.model_fields.keys()].to_dict(orient="records"):  # type: ignore[attr-defined]
        try:
            record = SleepRecord(**payload)
        except ValidationError as exc:  # pragma: no cover - surfaced in tests
            raise ValueError(f"Record failed schema validation: {exc}") from exc
        records.append(record)
    return records


def write_curated_dataset(df: pd.DataFrame, output_dir: Path) -> TransformResult:
    """Persist the curated dataset into a dated Parquet partition and bump the version pointer."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(tz=timezone.utc)
    partition_dir = output_dir / run_ts.strftime("%Y-%m-%d")
    partition_dir.mkdir(parents=True, exist_ok=True)
    part_name = run_ts.strftime("part-%H%M%S.parquet")
    part_path = partition_dir / part_name
    df.to_parquet(part_path, index=False)

    version_path = output_dir.parent / "VERSION.json"
    version_payload = {
        "last_run_ts": run_ts.isoformat(),
        "schema_hash": schema_hash(),
        "last_partition": str(part_path.relative_to(output_dir)),
    }
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(json.dumps(version_payload, indent=2))

    return TransformResult(dataframe=df, out_path=part_path, version_pointer=version_path)
