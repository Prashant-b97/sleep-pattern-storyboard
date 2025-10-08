"""Training orchestration for analytics v2 models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .detect import detect_anomalies, detect_change_points
from .features import build_circadian_features
from .models import ModelArtifact, gbm, prophet, sarimax, tft

MODEL_REGISTRY = {
    "sarimax": sarimax,
    "prophet": prophet,
    "gbm": gbm,
    "tft": tft,
}


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_curated_dataset(parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def _prepare_base_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "start_ts" not in raw_df.columns:
        raise ValueError("Curated dataset must include start_ts column.")
    # Build circadian features and aggregate by date
    feature_df = build_circadian_features(raw_df)
    feature_df["date"] = pd.to_datetime(feature_df["date"], utc=True)
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    return feature_df


def _prepare_target(df: pd.DataFrame, target_name: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    column = cfg.get("column", target_name)
    transform = cfg.get("transform")

    frame = df.copy()
    if column not in frame.columns:
        raise ValueError(f"Target column '{column}' not present in dataframe.")

    values = frame[column].astype(float)
    if transform == "minutes_to_hours":
        values = values / 60.0
    frame[target_name] = values

    numeric_cols = frame.select_dtypes(include=["number", "bool", "float", "int"]).columns
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    aggregated = frame.groupby("date")[numeric_cols].mean().reset_index()
    aggregated = aggregated.dropna(subset=[target_name])
    return aggregated


def run_training(config_path: Path, *, output_dir_override: Optional[Path] = None) -> List[ModelArtifact]:
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    parquet_dir = Path(data_cfg.get("parquet_dir", "data/processed/parquet"))

    raw_df = load_curated_dataset(parquet_dir)
    if raw_df.empty:
        raise ValueError(f"No curated parquet files found under {parquet_dir}.")

    base_frame = _prepare_base_frame(raw_df)

    reporting_cfg = config.get("reporting", {})
    output_dir = Path(reporting_cfg.get("output_dir", "analysis_output/model_reports"))
    if output_dir_override is not None:
        output_dir = output_dir_override
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_cfg = config.get("targets", {})
    models_cfg = config.get("models", {})

    artifacts: List[ModelArtifact] = []

    for target_name, target_cfg in targets_cfg.items():
        frame = _prepare_target(base_frame, target_name, target_cfg)
        if frame.empty:
            continue
        horizon = target_cfg.get("forecast_horizon", 7)
        backtest_window = target_cfg.get("backtest_window", 30)

        for model_name, model_cfg in models_cfg.items():
            if model_cfg is None:
                model_cfg = {}
            if not model_cfg.get("enabled", True):
                continue
            model_module = MODEL_REGISTRY.get(model_name)
            if model_module is None:
                continue

            cfg_kwargs = {k: v for k, v in model_cfg.items() if k != "enabled"}
            artifact = model_module.fit(
                frame,
                target=target_name,
                horizon=horizon,
                backtest_window=backtest_window,
                output_dir=output_dir,
                **cfg_kwargs,
            )

            if "actual" in artifact.forecast.columns:
                residuals = artifact.forecast["actual"] - artifact.forecast["prediction"]
            else:
                residuals = pd.Series(dtype=float)
            change_points = detect_change_points(frame[target_name])
            anomalies = detect_anomalies(residuals)

            artifact.diagnostics = {
                "change_points": change_points,
                "anomalies": anomalies.to_dict(orient="records"),
            }

            report_path = output_dir / f"{artifact.model_name}_{artifact.target}_report.json"
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        **artifact.to_dict(),
                        "change_points": change_points,
                        "anomalies": anomalies.to_dict(orient="records"),
                    },
                    fh,
                    indent=2,
                    default=str,
                )

            artifacts.append(artifact)

    summary_payload = {
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "config_path": str(config_path),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, default=str)

    return artifacts
