from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_system.models import TradeDecision
from trading_system.utils import ensure_dir


@dataclass(slots=True)
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: str
    trade_id: str
    decision_id: str | None
    symbol: str
    payload: dict[str, Any]


def stable_decision_id(decision: TradeDecision, decided_at: datetime) -> str:
    payload = "|".join(
        [
            decided_at.isoformat(),
            decision.symbol.upper(),
            decision.action.lower(),
            f"{decision.confidence:.6f}",
            f"{decision.allocation:.6f}",
        ]
    )
    return "dec_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_trade_id(symbol: str, side: str, entry_time: datetime, decision_id: str | None) -> str:
    payload = "|".join(
        [
            entry_time.isoformat(),
            symbol.upper(),
            side.lower(),
            decision_id or "",
        ]
    )
    return "trd_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_event_id(event_type: str, trade_id: str, timestamp: datetime | str, payload: dict[str, Any]) -> str:
    timestamp_text = timestamp if isinstance(timestamp, str) else timestamp.isoformat()
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return "evt_" + hashlib.sha256(
        f"{event_type}|{trade_id}|{timestamp_text}|{encoded}".encode("utf-8")
    ).hexdigest()[:16]


def append_audit_event(path: Path, event: AuditEvent) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), sort_keys=True, default=str))
        handle.write("\n")


def load_audit_events(path: Path) -> list[AuditEvent]:
    if not path.exists():
        return []
    events: list[AuditEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            events.append(AuditEvent(**json.loads(line)))
    return events
