from __future__ import annotations

from datetime import UTC, datetime

from trading_system.live_strategy_state import LivePositionState, LiveStrategyStateStore


def test_live_strategy_state_persists_partial_deferral_and_tax_notes(tmp_path):
    path = tmp_path / "state.json"
    store = LiveStrategyStateStore(path)
    position = LivePositionState(
        symbol="AAPL",
        side="long",
        quantity=10,
        entry_price=100.0,
        stop_price=95.0,
        take_profit_price=110.0,
        entry_timestamp="2026-01-02T14:30:00+00:00",
    )

    store.record_entry(position)
    store.mark_partial_exit("AAPL", quantity=5, new_stop_price=102.0)
    store.increment_thesis_deferral("AAPL")
    store.record_tax_cooldown("TSLA", blocked_until="2026-02-02T00:00:00+00:00")
    store.record_tax_reentry("TSLA", allowed_at="2026-01-15T14:30:00+00:00", size_multiplier=0.5)

    reloaded = LiveStrategyStateStore(path)

    assert reloaded.positions["AAPL"].partial_profit_taken is True
    assert reloaded.positions["AAPL"].partial_exit_quantity == 5
    assert reloaded.positions["AAPL"].stop_price == 102.0
    assert reloaded.positions["AAPL"].thesis_failure_deferrals == 1
    assert reloaded.tax_cooldowns["TSLA"]["blocked_until"] == "2026-02-02T00:00:00+00:00"
    assert reloaded.tax_reentries["TSLA"]["size_multiplier"] == 0.5


def test_live_strategy_state_reconciles_missing_positions(tmp_path):
    store = LiveStrategyStateStore(tmp_path / "state.json")
    store.record_entry(
        LivePositionState(
            symbol="AAPL",
            side="long",
            quantity=10,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            entry_timestamp=datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
        )
    )

    removed = store.reconcile_positions({"MSFT"})

    assert removed == ["AAPL"]
    assert store.positions == {}
