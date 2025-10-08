"""Utilities for privacy-aware data handling."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SALT_ENV = "SPS_SALT"
SENSITIVE_KEYWORDS = ("name", "email", "id")


def _get_salt(env_var: str = DEFAULT_SALT_ENV) -> str:
    salt = os.getenv(env_var, "sleep-pattern-salt")
    return salt


def _should_scrub(column: str, explicit: Iterable[str]) -> bool:
    lowered = column.lower()
    if lowered in {"user_id", "source"}:
        return False
    if lowered in (value.lower() for value in explicit):
        return True
    return any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)


def scrub_dataframe(df: pd.DataFrame, *, fields: Iterable[str], salt_env: str = DEFAULT_SALT_ENV) -> pd.DataFrame:
    """Return a copy of ``df`` with sensitive columns removed or hashed deterministically."""

    salt = _get_salt(salt_env)
    scrubbed = df.copy()
    for column in list(scrubbed.columns):
        if not _should_scrub(column, fields):
            continue
        if pd.api.types.is_numeric_dtype(scrubbed[column]):
            scrubbed = scrubbed.drop(columns=[column])
            continue
        scrubbed[column] = scrubbed[column].astype(str).apply(lambda value: _hash_value(value, salt))
    return scrubbed


def _hash_value(value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:16]


def ensure_local_path(path: Path, *, project_root: Path | None = None) -> Path:
    root = project_root or Path.cwd()
    resolved = path.resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError(f"Path {resolved} escapes project root {root}")
    return resolved
