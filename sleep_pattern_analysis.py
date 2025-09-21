#!/usr/bin/env python3
"""Compatibility wrapper around the packaged sleep analysis CLI."""
from __future__ import annotations

from sleep_analysis.cli import main as _main


def main() -> int:
    run_dir = _main()
    print(f"Analysis run completed. Outputs in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
