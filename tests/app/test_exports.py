from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import importlib.util
import sys

APP_ROOT = Path(__file__).resolve().parents[1].parent / "app"
if str(APP_ROOT.parent) not in sys.path:
    sys.path.append(str(APP_ROOT.parent))

spec = importlib.util.spec_from_file_location("app.components.tables", APP_ROOT / "components" / "tables.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)  # type: ignore[attr-defined]
export_dataframe = module.export_dataframe


def test_export_dataframe_creates_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(st, "success", lambda *args, **kwargs: None)
    df = pd.DataFrame({"sleep_hours": [7.5, 6.8], "efficiency": [0.82, 0.88]})
    export_dataframe(df, output_dir=tmp_path, basename="metrics", formats=["csv"])
    files = list(tmp_path.glob("metrics_*.csv"))
    assert files, "CSV export should create a file"
    assert files[0].stat().st_size > 0
