from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_system.utils import ensure_dir, read_json, write_json


@dataclass(slots=True)
class LivePositionState:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    stop_price: float | None
    take_profit_price: float | None
    entry_timestamp: str
    partial_profit_taken: bool = False
    partial_exit_quantity: int = 0
    thesis_failure_deferrals: int = 0
    latest_update_timestamp: str | None = None


class LiveStrategyStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.positions: dict[str, LivePositionState] = {}
        self.tax_cooldowns: dict[str, dict[str, Any]] = {}
        self.tax_reentries: dict[str, dict[str, Any]] = {}
        self._load()

    def record_entry(self, position: LivePositionState) -> None:
        position.latest_update_timestamp = _now()
        self.positions[position.symbol.upper()] = position
        self.save()

    def mark_partial_exit(self, symbol: str, *, quantity: int, new_stop_price: float | None) -> None:
        position = self.positions[symbol.upper()]
        position.partial_profit_taken = True
        position.partial_exit_quantity += quantity
        position.quantity = max(0, position.quantity - quantity)
        position.stop_price = new_stop_price
        position.latest_update_timestamp = _now()
        self.save()

    def increment_thesis_deferral(self, symbol: str) -> int:
        position = self.positions[symbol.upper()]
        position.thesis_failure_deferrals += 1
        position.latest_update_timestamp = _now()
        self.save()
        return position.thesis_failure_deferrals

    def record_tax_cooldown(self, symbol: str, *, blocked_until: str) -> None:
        self.tax_cooldowns[symbol.upper()] = {
            "symbol": symbol.upper(),
            "blocked_until": blocked_until,
            "recorded_at": _now(),
        }
        self.save()

    def record_tax_reentry(self, symbol: str, *, allowed_at: str, size_multiplier: float) -> None:
        self.tax_reentries[symbol.upper()] = {
            "symbol": symbol.upper(),
            "allowed_at": allowed_at,
            "size_multiplier": size_multiplier,
            "recorded_at": _now(),
        }
        self.save()

    def clear_position(self, symbol: str) -> None:
        self.positions.pop(symbol.upper(), None)
        self.save()

    def reconcile_positions(self, actual_symbols: set[str]) -> list[str]:
        normalized = {symbol.upper() for symbol in actual_symbols}
        removed = [symbol for symbol in self.positions if symbol not in normalized]
        for symbol in removed:
            del self.positions[symbol]
        if removed:
            self.save()
        return removed

    def save(self) -> None:
        write_json(
            self.path,
            {
                "positions": {
                    symbol: asdict(position) for symbol, position in self.positions.items()
                },
                "tax_cooldowns": self.tax_cooldowns,
                "tax_reentries": self.tax_reentries,
            },
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = read_json(self.path)
        self.positions = {
            symbol: LivePositionState(**position)
            for symbol, position in payload.get("positions", {}).items()
        }
        self.tax_cooldowns = payload.get("tax_cooldowns", {})
        self.tax_reentries = payload.get("tax_reentries", {})


def _now() -> str:
    return datetime.now(UTC).isoformat()
