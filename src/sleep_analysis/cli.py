#!/usr/bin/env python3
"""Command-line interface for the Sleep Pattern Analysis toolkit."""
from __future__ import annotations

import argparse
import logging
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

sns.set_theme(style="whitegrid")


@dataclass(slots=True)
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


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore sleep patterns from CSV exports.")
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


def main(args: Optional[list[str]] = None) -> Path:
    namespace = parse_args(args=args)
    return run_analysis(namespace)


if __name__ == "__main__":
    main()
