from pathlib import Path

import pandas as pd

from sleep_analysis.cli import load_primary, load_secondary


def test_load_primary_returns_sorted() -> None:
    df = load_primary(Path("sleep_data.csv"))
    assert list(df.columns) == ["date", "sleep_hours"]
    assert df["date"].is_monotonic_increasing
    assert df["sleep_hours"].dtype.kind in {"f", "i"}


def test_load_secondary_derives_hours() -> None:
    df = load_secondary(Path("sleep_data_new.csv"))
    if {"startDate", "endDate"}.issubset(df.columns):
        assert "hours" in df.columns
        assert pd.api.types.is_float_dtype(df["hours"])
