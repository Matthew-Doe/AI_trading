from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_system.utils import read_json, write_json


class TaxLossCooldownState:
    def __init__(self, path: Path):
        self.path = path
        self.cooldowns: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.cooldowns = {}
            return
        payload = read_json(self.path)
        self.cooldowns = payload.get("cooldowns", {}) if isinstance(payload, dict) else {}

    def save(self) -> None:
        write_json(
            self.path,
            {
                "updated_at": datetime.now(UTC).isoformat(),
                "cooldowns": self.cooldowns,
            },
        )

    def is_blocked(self, symbol: str, now: datetime | None = None) -> bool:
        blocked_until = self.blocked_until(symbol)
        if blocked_until is None:
            return False
        check_time = now or datetime.now(UTC)
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=UTC)
        if check_time < blocked_until:
            return True
        self.cooldowns.pop(symbol.upper(), None)
        self.save()
        return False

    def blocked_until(self, symbol: str) -> datetime | None:
        entry = self.cooldowns.get(symbol.upper())
        if not entry:
            return None
        value = entry.get("blocked_until")
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def record_loss_sale(
        self,
        *,
        symbol: str,
        sold_at: datetime | None = None,
        cooldown_days: int = 31,
        estimated_loss: float | None = None,
        source: str = "live_order_submission",
        notes: str | None = None,
    ) -> dict[str, Any]:
        loss_time = sold_at or datetime.now(UTC)
        if loss_time.tzinfo is None:
            loss_time = loss_time.replace(tzinfo=UTC)
        blocked_until = loss_time + timedelta(days=max(0, cooldown_days))
        payload = {
            "symbol": symbol.upper(),
            "last_loss_sale_at": loss_time.isoformat(),
            "blocked_until": blocked_until.isoformat(),
            "cooldown_days": max(0, cooldown_days),
            "estimated_loss": round(estimated_loss, 2) if estimated_loss is not None else None,
            "source": source,
            "notes": notes,
        }
        self.cooldowns[symbol.upper()] = payload
        self.save()
        return payload

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "cooldowns": self.cooldowns,
        }
