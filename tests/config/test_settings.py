from __future__ import annotations

from pathlib import Path

from sleep_analysis.config import load_settings


def test_load_settings_reads_yaml(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.yaml"
    settings_path.write_text(
        """
reports:
  output_dir: custom_reports
privacy:
  local_only: false
        """
    )
    settings = load_settings(settings_path)
    assert settings.reports_output_dir == Path("custom_reports")


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SPS_REPORTS__OUTPUT_DIR", "env_reports")
    settings = load_settings()
    assert settings.reports_output_dir == Path("env_reports")
    monkeypatch.delenv("SPS_REPORTS__OUTPUT_DIR")
