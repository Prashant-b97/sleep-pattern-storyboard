from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def parquet_selector(label: str, parquet_dir: Path) -> Optional[Path]:
    parquet_dir = parquet_dir.expanduser()
    files = sorted(parquet_dir.glob("**/*.parquet"))
    if not files:
        st.warning(f"No parquet files found under {parquet_dir}")
        return None
    options = {file.name: file for file in files}
    choice = st.selectbox(label, list(options.keys()))
    return options.get(choice)


def reports_selector(reports_dir: Path) -> list[Path]:
    reports_dir = reports_dir.expanduser()
    files = sorted(reports_dir.glob("*.json"))
    if not files:
        st.info("No model reports available yet.")
    return files
