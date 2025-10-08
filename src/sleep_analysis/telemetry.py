"""Simple telemetry logging utilities."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_OUTPUT = Path("analysis_output")


def log_run(
    event: str,
    *,
    start_time: float,
    rows_processed: int = 0,
    status: str = "success",
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> None:
    """Append a telemetry entry capturing duration and optional metadata."""

    duration_ms = int((time.time() - start_time) * 1000)
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "duration_ms": duration_ms,
        "rows_processed": rows_processed,
        "status": status,
    }
    if metadata:
        payload["metadata"] = metadata
    if error:
        payload["error"] = error

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "telemetry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
