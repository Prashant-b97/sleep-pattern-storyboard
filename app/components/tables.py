from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import streamlit as st


def model_metrics_table(reports_dir: Path) -> pd.DataFrame:
    records: List[dict] = []
    for path in sorted(reports_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", {})
        records.append(
            {
                "file": path.name,
                "model": payload.get("model_name") or payload.get("model"),
                "target": payload.get("target"),
                "mae": metrics.get("mae"),
                "mape": metrics.get("mape"),
                "coverage": metrics.get("coverage"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        st.dataframe(frame)
    else:
        st.info("No model reports found in analysis_output/model_reports/.")
    return frame


def export_dataframe(df: pd.DataFrame, *, output_dir: Path, basename: str, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if "csv" in formats:
        csv_path = output_dir / f"{basename}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        st.success(f"Exported CSV to {csv_path}")
    if "parquet" in formats:
        parquet_path = output_dir / f"{basename}_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)
        st.success(f"Exported Parquet to {parquet_path}")
