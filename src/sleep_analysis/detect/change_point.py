"""Change-point detection helpers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import ruptures as rpt  # type: ignore
except ImportError:  # pragma: no cover
    rpt = None  # type: ignore


def detect_change_points(
    series: pd.Series,
    *,
    model: str = "l1",
    penalty: float = 5.0,
    min_size: int = 5,
) -> List[int]:
    """Return indices where regime changes may have occurred."""

    cleaned = series.dropna().astype(float)
    if cleaned.empty or len(cleaned) < min_size * 2:
        return []

    if rpt is not None:
        algo = rpt.Pelt(model=model).fit(cleaned.values)
        return [int(cp) for cp in algo.predict(pen=penalty)[:-1] if cp < len(cleaned)]

    diffs = np.abs(np.diff(cleaned.values))
    threshold = penalty * np.std(diffs) if np.std(diffs) > 0 else penalty
    return [idx + 1 for idx, value in enumerate(diffs) if value >= threshold]
