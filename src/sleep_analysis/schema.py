"""Unified schema definitions for curated sleep records."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SleepRecord(BaseModel):
    """Canonical representation of a single sleep record across vendors."""

    id: str
    user_id: str
    start_ts: datetime = Field(..., description="Sleep start timestamp (timezone aware).")
    end_ts: datetime = Field(..., description="Sleep end timestamp (timezone aware).")
    duration_min: float = Field(..., gt=0, description="Sleep duration in minutes.")
    efficiency: float = Field(..., ge=0.0, le=1.0, description="Sleep efficiency ratio.")
    is_nap: bool
    tz: str = Field(..., description="IANA timezone string used for tracking.")
    source: str = Field(..., description="Origin vendor identifier.")

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("start_ts", "end_ts")
    @staticmethod
    def validate_timezone(value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return value

    @field_validator("end_ts")
    @staticmethod
    def validate_end(value: datetime, info) -> datetime:
        start: datetime = info.data.get("start_ts")  # type: ignore[attr-defined]
        if start and value <= start:
            raise ValueError("end_ts must be after start_ts.")
        return value


def schema_hash() -> str:
    """Return a stable hash of the SleepRecord schema for version tracking."""
    schema_json = json.dumps(SleepRecord.model_json_schema(), sort_keys=True)
    return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
