from __future__ import annotations

from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.8 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore

import pytest
from hypothesis import given, strategies as st

from sleep_analysis.schema import SleepRecord


@st.composite
def sleep_payloads(draw: st.DrawFn) -> dict[str, object]:
    tz_name = draw(st.sampled_from(["UTC", "America/Los_Angeles", "Europe/London"]))
    tzinfo = ZoneInfo(tz_name)
    base_start = draw(
        st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2024, 12, 31),
        )
    ).replace(tzinfo=timezone.utc).astimezone(tzinfo)
    duration = draw(
        st.timedeltas(
            min_value=timedelta(minutes=30),
            max_value=timedelta(hours=12),
        )
    )
    end = base_start + duration
    return {
        "id": draw(st.text(min_size=3, max_size=12)),
        "user_id": draw(st.text(min_size=3, max_size=12)),
        "start_ts": base_start,
        "end_ts": end,
        "duration_min": duration.total_seconds() / 60,
        "efficiency": draw(st.floats(min_value=0.0, max_value=1.0)),
        "is_nap": draw(st.booleans()),
        "tz": tz_name,
        "source": draw(st.sampled_from(["oura", "fitbit", "apple_health"])),
    }


@given(sleep_payloads())
def test_sleep_record_accepts_valid_payload(payload: dict[str, object]) -> None:
    record = SleepRecord(**payload)
    assert record.duration_min > 0
    assert 0 <= record.efficiency <= 1
    assert record.end_ts > record.start_ts


def test_sleep_record_rejects_invalid_efficiency() -> None:
    payload = {
        "id": "bad",
        "user_id": "user",
        "start_ts": datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc),
        "end_ts": datetime(2024, 1, 2, 6, 0, tzinfo=timezone.utc),
        "duration_min": 480,
        "efficiency": 1.4,
        "is_nap": False,
        "tz": "UTC",
        "source": "test",
    }
    with pytest.raises(ValueError):
        SleepRecord(**payload)


def test_sleep_record_requires_timezone_awareness() -> None:
    payload = {
        "id": "bad",
        "user_id": "user",
        "start_ts": datetime(2024, 1, 1, 22, 0),
        "end_ts": datetime(2024, 1, 2, 6, 0),
        "duration_min": 480,
        "efficiency": 0.9,
        "is_nap": False,
        "tz": "UTC",
        "source": "test",
    }
    with pytest.raises(ValueError):
        SleepRecord(**payload)
