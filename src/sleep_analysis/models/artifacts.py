"""Common data structures for model training artefacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ModelArtifact:
    """Container describing the output of a model training run."""

    model_name: str
    target: str
    metrics: Dict[str, float]
    model_path: Optional[str]
    forecast_path: str
    forecast: pd.DataFrame = field(repr=False)
    diagnostics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "target": self.target,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "forecast_path": self.forecast_path,
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        return payload
