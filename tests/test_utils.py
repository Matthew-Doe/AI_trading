from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, date

from trading_system.utils import dataclass_to_dict


@dataclass
class Inner:
    created_at: datetime
    trade_date: date


@dataclass
class Outer:
    inner: Inner


def test_dataclass_to_dict_serializes_nested_datetime_and_date():
    payload = Outer(
        inner=Inner(
            created_at=datetime(2026, 4, 26, 1, 45, 13, tzinfo=UTC),
            trade_date=date(2026, 3, 2),
        )
    )

    result = dataclass_to_dict(payload)

    assert result == {
        "inner": {
            "created_at": "2026-04-26T01:45:13+00:00",
            "trade_date": "2026-03-02",
        }
    }
