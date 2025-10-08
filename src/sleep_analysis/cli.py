#!/usr/bin/env python3
"""Command-line interface for the Sleep Pattern Analysis toolkit."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import statsmodels.api as sm
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:  # pragma: no cover
    sm = None
    AutoReg = None
    seasonal_decompose = None

from .config import load_settings
from .parsers import parse_apple_health, parse_fitbit, parse_oura
from .transform import normalize, validate_records, write_curated_dataset
from .training import run_training
from .features.health_store import build_health_feature_store
from .telemetry import log_run
from .privacy.scrub import scrub_dataframe, ensure_local_path, DEFAULT_SALT_ENV
from .reporting.pdf import generate_weekly_pdf

sns.set_theme(style="whitegrid")

SETTINGS = load_settings()


@dataclass
class RunConfig:
    primary_path: Path
    secondary_path: Optional[Path]
    output_base: Path
    run_dir: Path
    figures_dir: Path
    tables_dir: Path
    lags: int
    forecast_horizon: int
    seasonal_period: int


COMMAND_ALIASES = {"analyze", "ingest", "train", "build-features", "export-report"}


def _configure_analyze_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--primary",
        default="data/raw/sleep_data.csv",
        help="Daily summary CSV path",
    )
    parser.add_argument(
        "--secondary",
        default="data/raw/sleep_data_new.csv",
        help="Event-level CSV path (set --skip-secondary to ignore)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Base directory where run artefacts are stored",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. Defaults to timestamp: run_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=30,
        help="Number of lags for autocorrelation and AutoReg modelling",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=14,
        help="Forecast horizon (days) for the AutoReg model",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=7,
        help="Seasonal period (days) for decomposition",
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Skip secondary dataset processing even if the file exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep Pattern Storyboard CLI")
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run exploratory analysis from Act I",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _configure_analyze_parser(analyze_parser)
    analyze_parser.set_defaults(command="analyze")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest vendor CSVs into the curated parquet store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ingest_parser.add_argument(
        "--source",
        required=True,
        choices=["oura", "fitbit", "apple_health"],
        help="Vendor source identifier",
    )
    ingest_parser.add_argument(
        "--path",
        required=True,
        help="Path to raw CSV file",
    )
    ingest_parser.add_argument(
        "--out",
        default="data/processed/parquet",
        help="Directory where curated parquet partitions are written",
    )
    ingest_parser.add_argument(
        "--privacy",
        choices=["local-only"],
        default=None,
        help="Enable additional privacy safeguards for this run",
    )
    ingest_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for ingest",
    )
    ingest_parser.set_defaults(command="ingest")

    features_parser = subparsers.add_parser(
        "build-features",
        help="Fuse health signals into a feature store parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    features_parser.add_argument(
        "--parquet-dir",
        default="data/processed/parquet",
        help="Directory containing curated sleep parquet partitions",
    )
    features_parser.add_argument("--hrv", default=None, help="Path to HRV CSV export")
    features_parser.add_argument("--steps", default=None, help="Path to steps CSV export")
    features_parser.add_argument("--tags", default=None, help="Path to manual tags CSV")
    features_parser.add_argument(
        "--screen-time",
        dest="screen_time",
        default=None,
        help="Path to screen-time proxy CSV",
    )
    features_parser.add_argument(
        "--out",
        required=True,
        help="Directory where feature store parquet files are written",
    )
    features_parser.add_argument(
        "--privacy",
        choices=["local-only"],
        default=None,
        help="Enable additional privacy safeguards for this run",
    )
    features_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for feature building",
    )
    features_parser.set_defaults(command="build-features")

    export_parser = subparsers.add_parser(
        "export-report",
        help="Generate a weekly PDF report from analysis outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export_parser.add_argument(
        "--week",
        default=None,
        help="ISO week (YYYY-Www) to render. Defaults to latest available week.",
    )
    export_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for report export",
    )
    export_parser.set_defaults(command="export-report")

    train_parser = subparsers.add_parser(
        "train",
        help="Train analytics v2 models using the provided config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to analytics v2 YAML configuration",
    )
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override report output directory",
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for training",
    )
    train_parser.set_defaults(command="train")

    return parser


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    if args is None:
        args = sys.argv[1:]
    parser = build_parser()
    if args and args[0] in {"-h", "--help"}:
        return parser.parse_args(args=args)
    if not args:
        args = ["analyze"]
    elif args[0] not in COMMAND_ALIASES:
        if args[0].startswith("-"):
            args = ["analyze", *args]
        else:
            args = ["analyze", *args]
    return parser.parse_args(args=args)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_dirs(base: Path, run_id: Optional[str]) -> tuple[Path, Path, Path]:
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / run_id
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return run_dir, figures, tables


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "metric": ["rows", "columns", "duplicates"],
            "value": [len(df), df.shape[1], int(df.duplicated().sum())],
        }
    )
    na_counts = df.isnull().sum()
    if na_counts.any():
        na_df = na_counts.reset_index()
        na_df.columns = ["column", "missing"]
        summary = pd.concat([summary, na_df], ignore_index=True)
    return summary


def save_plot(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def plot_primary_timeseries(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(ax=ax, data=df, x="date", y="sleep_hours", linewidth=1.5)
    ax.set_title("Sleep Hours Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours Slept")
    fig.autofmt_xdate()
    save_plot(fig, output / "primary_sleep_hours_line.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(ax=ax, data=df, x="date", y="sleep_hours", s=20)
    ax.set_title("Sleep Hours Scatter")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours Slept")
    fig.autofmt_xdate()
    save_plot(fig, output / "primary_sleep_hours_scatter.png")


def plot_primary_distributions(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ax=ax, data=df, x="sleep_hours", bins=30, kde=True, color="steelblue")
    ax.set_title("Sleep Hours Histogram")
    ax.set_xlabel("Hours Slept")
    save_plot(fig, output / "primary_sleep_hours_hist.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(ax=ax, data=df, x="sleep_hours", fill=True, color="darkorange")
    ax.set_title("Sleep Hours Density")
    ax.set_xlabel("Hours Slept")
    save_plot(fig, output / "primary_sleep_hours_density.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(ax=ax, data=df, x="sleep_hours", color="lightgreen")
    ax.set_title("Sleep Hours Boxplot")
    ax.set_xlabel("Hours Slept")
    save_plot(fig, output / "primary_sleep_hours_boxplot.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(ax=ax, data=df, x="sleep_hours", inner="quartile", color="plum")
    ax.set_title("Sleep Hours Violin Plot")
    ax.set_xlabel("Hours Slept")
    save_plot(fig, output / "primary_sleep_hours_violin.png")


def plot_secondary_distributions(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(ax=ax, data=df, x="hours", bins=24, kde=True, color="slateblue")
    ax.set_title("Sleep Session Duration Histogram")
    ax.set_xlabel("Duration (hours)")
    save_plot(fig, output / "secondary_hours_hist.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(ax=ax, data=df, x="hours", fill=True, color="salmon")
    ax.set_title("Sleep Session Duration Density")
    ax.set_xlabel("Duration (hours)")
    save_plot(fig, output / "secondary_hours_density.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(ax=ax, data=df, x="creationDate", y="hours", linewidth=1.5)
    ax.set_title("Sleep Session Duration Over Time")
    ax.set_xlabel("Creation Date")
    ax.set_ylabel("Duration (hours)")
    fig.autofmt_xdate()
    save_plot(fig, output / "secondary_hours_line.png")

    if "sourceName" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(
            ax=ax,
            data=df,
            x="sourceName",
            y="hours",
            estimator="mean",
            errorbar=None,
        )
        ax.set_title("Average Sleep Duration by Source")
        ax.set_xlabel("Source")
        ax.set_ylabel("Average Duration (hours)")
        ax.tick_params(axis="x", rotation=60, labelsize=9)
        save_plot(fig, output / "secondary_hours_by_source.png")

    if "device" in df.columns and df["device"].notna().any():
        device_counts = df["device"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(
            device_counts.values,
            labels=device_counts.index,
            autopct="%1.1f%%",
            textprops={"fontsize": 12},
        )
        ax.set_title("Device Distribution")
        save_plot(fig, output / "secondary_device_distribution.png")


def plot_lag_and_acf(series: pd.Series, output: Path, lags: int) -> None:
    if sm is None:
        logging.warning("statsmodels is not installed; skipping autocorrelation diagnostics")
        return

    fig = sm.graphics.tsa.plot_acf(series, lags=min(lags, len(series) - 1))
    fig.suptitle("Autocorrelation Function")
    save_plot(fig, output / "primary_sleep_hours_acf.png")

    fig = sm.graphics.tsa.plot_acf(series, lags=min(lags, len(series) - 1))
    fig.suptitle("Autocorrelation Function (zoom)")
    save_plot(fig, output / "primary_sleep_hours_acf_zoom.png")

    from pandas.plotting import lag_plot

    fig, ax = plt.subplots(figsize=(6, 6))
    lag_plot(series, ax=ax)
    ax.set_title("Lag Plot")
    save_plot(fig, output / "primary_sleep_hours_lag_plot.png")


def run_autoreg(series: pd.Series, tables_dir: Path, figures_dir: Path, lags: int, forecast_horizon: int) -> pd.DataFrame:
    if AutoReg is None:
        logging.warning("statsmodels is not installed; skipping AutoReg modelling")
        return pd.DataFrame()
    if len(series) <= lags:
        logging.warning("Series length is insufficient for AutoReg with %s lags", lags)
        return pd.DataFrame()

    series = series.asfreq("D")
    if series.isna().any():
        series = series.interpolate(limit_direction="both")

    logging.info("Fitting AutoReg(lag=%s) model", lags)
    model = AutoReg(series, lags=lags, old_names=False)
    model_fit = model.fit()
    params = model_fit.params.rename("coefficient")
    params.to_frame().to_csv(tables_dir / "primary_autoreg_coefficients.csv")
    logging.info("AutoReg fitted with %s coefficients", len(params))

    fitted = model_fit.fittedvalues
    comparison = pd.DataFrame(
        {
            "date": fitted.index,
            "actual": series.loc[fitted.index],
            "fitted": fitted,
        }
    ).reset_index(drop=True)
    comparison.to_csv(tables_dir / "primary_autoreg_fitted_vs_actual.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, label="Actual", linewidth=1.5)
    ax.plot(fitted.index, fitted.values, label="Fitted", linewidth=1.5)
    ax.set_title("AutoReg Fitted vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours Slept")
    ax.legend()
    fig.autofmt_xdate()
    save_plot(fig, figures_dir / "primary_autoreg_fitted_vs_actual.png")

    forecast = model_fit.forecast(steps=forecast_horizon)
    forecast_index = pd.date_range(start=fitted.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    forecast_df = pd.DataFrame({"date": forecast_index, "forecast_hours": forecast.values})
    forecast_df.to_csv(tables_dir / "primary_autoreg_forecast.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series.index, series.values, label="History", linewidth=1.5)
    ax.plot(forecast_index, forecast.values, label="Forecast", linewidth=1.5)
    ax.set_title("AutoReg Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hours Slept")
    ax.legend()
    fig.autofmt_xdate()
    save_plot(fig, figures_dir / "primary_autoreg_forecast.png")

    return forecast_df


def run_seasonal_decomposition(
    series: pd.Series, figures_dir: Path, tables_dir: Path, seasonal_period: int
) -> Optional[pd.DataFrame]:
    if seasonal_decompose is None:
        logging.warning("statsmodels is not installed; skipping seasonal decomposition")
        return None
    if len(series) < seasonal_period * 2:
        logging.warning("Series length is insufficient for seasonal decomposition with period %s", seasonal_period)
        return None

    series = series.asfreq("D")
    if series.isna().any():
        series = series.interpolate(limit_direction="both")

    result = seasonal_decompose(series, period=seasonal_period, model="additive")

    fig = result.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle("Seasonal Decomposition of Sleep Hours")
    save_plot(fig, figures_dir / "primary_sleep_hours_decomposition.png")

    components = pd.DataFrame(
        {
            "date": series.index,
            "observed": result.observed.values,
            "trend": result.trend.values,
            "seasonal": result.seasonal.values,
            "resid": result.resid.values,
        }
    )
    components.to_csv(tables_dir / "primary_sleep_hours_decomposition.csv", index=False)
    return components


def load_primary(path: Path) -> pd.DataFrame:
    logging.info("Loading primary dataset: %s", path)
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the primary dataset")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    df = df.dropna(subset=["date", "sleep_hours"]).sort_values("date")
    return df


def load_secondary(path: Path) -> pd.DataFrame:
    logging.info("Loading secondary dataset: %s", path)
    df = pd.read_csv(path)
    datetime_cols = ["creationDate", "startDate", "endDate"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    if "startDate" in df.columns and "endDate" in df.columns:
        df["Duration"] = df["endDate"] - df["startDate"]
        df["hours"] = df["Duration"].dt.total_seconds() / 3600
    return df


def export_tables(df: pd.DataFrame, tables_dir: Path, prefix: str) -> None:
    describe_dataframe(df).to_csv(tables_dir / f"{prefix}_summary.csv", index=False)
    df.describe(include="all").transpose().to_csv(tables_dir / f"{prefix}_describe.csv")


def build_run_config(args: argparse.Namespace) -> RunConfig:
    primary_path = Path(args.primary)
    if not primary_path.exists():
        raise SystemExit(f"Primary dataset not found: {primary_path}")

    secondary_path = Path(args.secondary)
    if args.skip_secondary:
        secondary_path = None
    elif not secondary_path.exists():
        logging.warning("Secondary dataset not found: %s", secondary_path)
        secondary_path = None

    output_base = Path(args.output_dir)
    run_dir, figures_dir, tables_dir = ensure_dirs(output_base, args.run_id)

    return RunConfig(
        primary_path=primary_path,
        secondary_path=secondary_path,
        output_base=output_base,
        run_dir=run_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        lags=args.lags,
        forecast_horizon=args.forecast_horizon,
        seasonal_period=args.seasonal_period,
    )


def summarise_run(run_config: RunConfig, forecast: pd.DataFrame | None, decomposition: pd.DataFrame | None) -> None:
    lines = ["Sleep Pattern Analysis Run Summary", "=" * 34, ""]
    lines.append(f"Run directory: {run_config.run_dir}")
    lines.append(f"Primary dataset: {run_config.primary_path}")
    if run_config.secondary_path:
        lines.append(f"Secondary dataset: {run_config.secondary_path}")
    lines.append(f"AutoReg lags: {run_config.lags}")
    lines.append(f"Forecast horizon: {run_config.forecast_horizon} days")
    lines.append(f"Seasonal period: {run_config.seasonal_period} days")
    lines.append("")

    if forecast is not None and not forecast.empty:
        next_point = forecast.iloc[0]
        lines.append(
            f"Next-day forecast: {next_point['date'].date()} -> {next_point['forecast_hours']:.2f} hours"
        )
        lines.append("")

    if decomposition is not None:
        trend_mean = decomposition["trend"].mean(skipna=True)
        seasonal_amp = decomposition["seasonal"].abs().mean(skipna=True) * 2
        lines.append(f"Average trend level: {trend_mean:.2f} hours")
        lines.append(f"Approx seasonal swing: ±{seasonal_amp/2:.2f} hours")
        lines.append("")

    summary_path = run_config.run_dir / "run_summary.txt"
    summary_path.write_text("\n".join(lines))


def run_analysis(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)
    config = build_run_config(args)

    primary_df = load_primary(config.primary_path)
    export_tables(primary_df, config.tables_dir, "primary")

    plot_df = primary_df.copy()
    plot_primary_timeseries(plot_df, config.figures_dir)
    plot_primary_distributions(plot_df, config.figures_dir)

    sleep_series = primary_df.set_index("date")["sleep_hours"]

    plot_lag_and_acf(sleep_series, config.figures_dir, config.lags)
    forecast_df = run_autoreg(
        sleep_series,
        tables_dir=config.tables_dir,
        figures_dir=config.figures_dir,
        lags=config.lags,
        forecast_horizon=config.forecast_horizon,
    )

    decomposition_df = run_seasonal_decomposition(
        sleep_series,
        figures_dir=config.figures_dir,
        tables_dir=config.tables_dir,
        seasonal_period=config.seasonal_period,
    )

    if config.secondary_path is not None:
        secondary_df = load_secondary(config.secondary_path)
        if "hours" in secondary_df.columns:
            export_tables(secondary_df, config.tables_dir, "secondary")
            plot_secondary_distributions(secondary_df, config.figures_dir)
        else:
            logging.warning("Secondary dataset missing duration columns; skipping plots")

    summarise_run(config, forecast_df, decomposition_df)
    logging.info("Analysis complete. Outputs saved under %s", config.run_dir)
    return config.run_dir


def run_ingest(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)
    start = time.time()
    parser_map = {
        "oura": parse_oura,
        "fitbit": parse_fitbit,
        "apple_health": parse_apple_health,
    }
    parser_fn = parser_map[args.source]
    logging.info("Parsing %s export from %s", args.source, args.path)
    raw_df = parser_fn(args.path)
    if raw_df.empty:
        logging.warning("No rows detected in input %s", args.path)
    normalized = normalize(raw_df)
    validate_records(normalized)
    privacy_mode = args.privacy or ("local-only" if SETTINGS.privacy_local_only else None)
    scrub_fields = SETTINGS.get("privacy", "scrub_fields", default=[])
    salt_env = SETTINGS.get("privacy", "hash_salt_env", default=DEFAULT_SALT_ENV)
    out_dir = Path(args.out)
    if privacy_mode == "local-only":
        out_dir = Path(ensure_local_path(out_dir))
        normalized = scrub_dataframe(normalized, fields=scrub_fields, salt_env=salt_env)
    result = write_curated_dataset(normalized, out_dir)
    logging.info("Wrote %d rows to %s", len(normalized), result.out_path)
    logging.info("Version pointer updated at %s", result.version_pointer)
    log_run(
        "ingest",
        start_time=start,
        rows_processed=len(normalized),
        metadata={
            "source": args.source,
            "output": str(result.out_path),
            "privacy": privacy_mode or ("local-only" if SETTINGS.privacy_local_only else "default"),
        },
    )
    return result.out_path


def run_build_features(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)
    start = time.time()
    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.out)
    privacy_mode = args.privacy or ("local-only" if SETTINGS.privacy_local_only else None)
    scrub_fields = SETTINGS.get("privacy", "scrub_fields", default=[])
    salt_env = SETTINGS.get("privacy", "hash_salt_env", default=DEFAULT_SALT_ENV)
    if privacy_mode == "local-only":
        output_dir = Path(ensure_local_path(output_dir))
    feature_path = build_health_feature_store(
        parquet_dir,
        hrv_path=Path(args.hrv) if args.hrv else None,
        steps_path=Path(args.steps) if args.steps else None,
        tags_path=Path(args.tags) if args.tags else None,
        screen_time_path=Path(args.screen_time) if args.screen_time else None,
        output_dir=output_dir,
        scrub_fields=scrub_fields if privacy_mode == "local-only" else None,
        local_only=privacy_mode == "local-only",
        salt_env=salt_env,
    )
    logging.info("Feature store written to %s", feature_path)
    schema_doc = output_dir / "SCHEMA.json"
    rows = 0
    if schema_doc.exists():
        try:
            import json

            rows = json.loads(schema_doc.read_text()).get("rows", 0)
        except Exception:  # pragma: no cover - telemetry best-effort
            rows = 0
    log_run(
        "build_features",
        start_time=start,
        rows_processed=int(rows),
        metadata={
            "output": str(feature_path),
            "hrv": args.hrv,
            "steps": args.steps,
            "tags": args.tags,
            "screen_time": args.screen_time,
            "privacy": privacy_mode or ("local-only" if SETTINGS.privacy_local_only else "default"),
        },
    )
    return feature_path


def run_train(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)
    start = time.time()
    config_path = Path(args.config)
    output_override = Path(args.output_dir) if args.output_dir else None
    artifacts = run_training(config_path, output_dir_override=output_override)
    if artifacts:
        summary_dir = Path(args.output_dir) if args.output_dir else Path(artifacts[0].forecast_path).parent
        logging.info("Training completed, produced %d artefacts.", len(artifacts))
        rows = sum(len(artifact.forecast) for artifact in artifacts if artifact.forecast is not None)
        log_run(
            "train",
            start_time=start,
            rows_processed=int(rows),
            metadata={"artifacts": len(artifacts), "config": str(config_path)},
        )
        return summary_dir
    logging.warning("No artefacts generated during training run.")
    log_run(
        "train",
        start_time=start,
        status="error",
        error="no artifacts",
        metadata={"config": str(config_path)},
    )
    return Path(args.output_dir or ".")


def run_export_report(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)
    start = time.time()
    reports_dir = Path(SETTINGS.get("app", "model_reports_dir", default="analysis_output/model_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = SETTINGS.reports_output_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = generate_weekly_pdf(reports_dir, pdf_dir, week=args.week)
    log_run(
        "export_report",
        start_time=start,
        metadata={"week": args.week or "latest", "path": str(pdf_path)},
    )
    return pdf_path


def main(args: Optional[list[str]] = None) -> Path:
    namespace = parse_args(args=args)
    if namespace.command == "ingest":
        return run_ingest(namespace)
    if namespace.command == "train":
        return run_train(namespace)
    if namespace.command == "build-features":
        return run_build_features(namespace)
    if namespace.command == "export-report":
        return run_export_report(namespace)
    return run_analysis(namespace)


if __name__ == "__main__":
    main()
