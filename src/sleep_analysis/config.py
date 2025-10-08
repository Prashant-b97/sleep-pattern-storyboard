"""Configuration loader handling YAML settings with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

SETTINGS_PATH = Path("config/settings.yaml")
ENV_PREFIX = "SPS_"


@dataclass
class Settings:
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.raw
        for key in keys:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                return default
        return cursor

    @property
    def privacy_local_only(self) -> bool:
        return bool(self.get("privacy", "local_only", default=False))

    @property
    def reports_output_dir(self) -> Path:
        return Path(self.get("reports", "output_dir", default="analysis_output/reports"))

    @property
    def exports_output_dir(self) -> Path:
        return Path(self.get("exports", "output_dir", default="analysis_output/exports"))


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_env_overrides(settings: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        cursor = overrides
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[path[-1]] = value

    def merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = merge(base[key], value)
            else:
                base[key] = _coerce(value)
        return base

    return merge(settings, overrides)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_settings(path: Path | None = None) -> Settings:
    settings_path = path or SETTINGS_PATH
    raw = _read_yaml(settings_path)
    raw = _apply_env_overrides(raw)
    return Settings(raw=raw)
