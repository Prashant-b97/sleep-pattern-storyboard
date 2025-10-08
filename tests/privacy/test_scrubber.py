from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sleep_analysis.privacy.scrub import DEFAULT_SALT_ENV, ensure_local_path, scrub_dataframe


def test_scrub_dataframe_hashes_sensitive_fields(monkeypatch) -> None:
    monkeypatch.setenv(DEFAULT_SALT_ENV, "unit-test-salt")
    df = pd.DataFrame(
        {
            "user_id": ["user123"],
            "name": ["Test User"],
            "email": ["user@example.com"],
            "duration_min": [420],
        }
    )
    scrubbed = scrub_dataframe(df, fields=["name", "email"])
    assert scrubbed["user_id"].iloc[0] == "user123"
    assert scrubbed["name"].iloc[0] != "Test User"
    scrubbed_again = scrub_dataframe(df, fields=["name", "email"])
    assert scrubbed_again["name"].iloc[0] == scrubbed["name"].iloc[0]


def test_ensure_local_path(tmp_path: Path) -> None:
    inside = tmp_path / "subdir" / "file.txt"
    inside.parent.mkdir(parents=True, exist_ok=True)
    resolved = ensure_local_path(inside, project_root=tmp_path)
    assert resolved == inside.resolve()
    outside = Path("/tmp/outside.txt")
    with pytest.raises(ValueError):
        ensure_local_path(outside, project_root=tmp_path)
