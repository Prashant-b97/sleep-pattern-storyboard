"""Residual anomaly detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(residuals: pd.Series, threshold: float) -> pd.Series:
    mu = residuals.mean()
    sigma = residuals.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(False, index=residuals.index)
    return residuals.sub(mu).abs().ge(threshold * sigma)


def _iqr(residuals: pd.Series, threshold: float) -> pd.Series:
    q1 = residuals.quantile(0.25)
    q3 = residuals.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return pd.Series(False, index=residuals.index)
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return (residuals < lower) | (residuals > upper)


def detect_anomalies(
    residuals: pd.Series,
    *,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Return dataframe with anomaly flags based on residual signals."""

    residuals = residuals.dropna()
    if residuals.empty:
        return pd.DataFrame({"residual": [], "is_anomaly": []})

    if method == "iqr":
        mask = _iqr(residuals, threshold)
    else:
        mask = _zscore(residuals, threshold)

    frame = pd.DataFrame(
        {
            "residual": residuals,
            "is_anomaly": mask,
            "method": method,
            "threshold": threshold,
        }
    )
    return frame
