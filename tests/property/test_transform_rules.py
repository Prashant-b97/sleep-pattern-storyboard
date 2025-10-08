from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.8 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from sleep_analysis.transform import normalize, validate_records, write_curated_dataset


@st.composite
def raw_frames(draw: st.DrawFn) -> pd.DataFrame:
    size = draw(st.integers(min_value=3, max_value=10))
    rows: list[dict[str, object]] = []
    tz_choices = ["UTC", "America/Los_Angeles", "Europe/London"]
    for idx in range(size):
        tz_name = draw(st.sampled_from(tz_choices))
        tzinfo = ZoneInfo(tz_name)
        start_aware = (
            draw(
                st.datetimes(
                    min_value=datetime(2023, 1, 1),
                    max_value=datetime(2024, 12, 31),
                )
            )
            .replace(tzinfo=timezone.utc)
            .astimezone(tzinfo)
        )
        duration = draw(
            st.timedeltas(min_value=timedelta(minutes=10), max_value=timedelta(hours=15))
        )
        end_aware = start_aware + duration
        # Drop tz info to ensure normalize() reinstates it.
        start_naive = start_aware.replace(tzinfo=None)
        end_naive = end_aware.replace(tzinfo=None)
        rows.append(
            {
                "id": f"rec_{idx}",
                "user_id": "user123",
                "start_ts": pd.Timestamp(start_naive),
                "end_ts": pd.Timestamp(end_naive),
                "duration_min": duration.total_seconds() / 60,
                "efficiency": draw(st.floats(min_value=0.0, max_value=1.0)),
                "is_nap": draw(st.booleans()),
                "tz": tz_name,
                "source": draw(st.sampled_from(["oura", "fitbit", "apple_health"])),
            }
        )
    return pd.DataFrame(rows)


@settings(max_examples=25)
@given(raw_frames())
def test_normalize_clips_and_flags(raw: pd.DataFrame) -> None:
    normalized = normalize(raw)
    assert normalized["duration_min"].between(120, 720).all()
    assert normalized["start_ts"].dropna().map(lambda ts: ts.tzinfo is not None).all()
    assert normalized["end_ts"].dropna().map(lambda ts: ts.tzinfo is not None).all()
    assert {"duration_zscore", "is_outlier_duration", "is_weekend"}.issubset(
        normalized.columns
    )
    # Validate schema compliance
    records = list(validate_records(normalized))
    assert len(records) == len(normalized)


def test_write_curated_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["a"],
            "user_id": ["u"],
            "start_ts": [pd.Timestamp("2024-01-01T23:00:00-08:00")],
            "end_ts": [pd.Timestamp("2024-01-02T06:00:00-08:00")],
            "duration_min": [480.0],
            "efficiency": [0.9],
            "is_nap": [False],
            "tz": ["America/Los_Angeles"],
            "source": ["oura"],
        }
    )
    normalized = normalize(df)
    result = write_curated_dataset(normalized, tmp_path / "parquet")
    assert result.out_path.exists()
    pointer_path = tmp_path / "VERSION.json"
    assert pointer_path.exists()
    payload = pd.read_json(pointer_path, typ="series")
    assert "schema_hash" in payload.index
