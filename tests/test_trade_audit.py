from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_system.backtest_execution import BacktestExecutionEngine
from trading_system.config import TradingConfig
from trading_system.models import TradeDecision
from trading_system.trade_audit import (
    AuditEvent,
    append_audit_event,
    load_audit_events,
    stable_decision_id,
    stable_trade_id,
)


def test_stable_ids_are_repeatable_and_distinguish_inputs():
    decided_at = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    decision = TradeDecision(symbol="AAPL", action="long", confidence=0.82, allocation=0.1)

    first = stable_decision_id(decision, decided_at)
    second = stable_decision_id(decision, decided_at)
    changed = stable_decision_id(
        TradeDecision(symbol="AAPL", action="short", confidence=0.82, allocation=0.1),
        decided_at,
    )

    assert first == second
    assert first != changed
    assert stable_trade_id("AAPL", "long", decided_at, first) == stable_trade_id(
        "AAPL", "long", decided_at, first
    )


def test_audit_event_jsonl_round_trip(tmp_path):
    path = tmp_path / "events.jsonl"
    event = AuditEvent(
        event_id="evt-1",
        event_type="entry_filled",
        timestamp="2026-01-02T14:30:00+00:00",
        trade_id="trade-1",
        decision_id="decision-1",
        symbol="AAPL",
        payload={"qty": 10, "fill_price": 101.5},
    )

    append_audit_event(path, event)

    assert load_audit_events(path) == [event]


def test_backtest_writes_entry_partial_and_full_audit_events(tmp_path):
    config = TradingConfig(enable_partial_profit_taking=True, partial_profit_take_fraction=0.50)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)
    engine.audit_event_path = tmp_path / "trade_audit_events.jsonl"
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=110.0,
                invalidation_price=90.0,
            )
        ],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions([], {"AAPL": 112.0}, start + timedelta(days=1))
    engine.process_decisions([], {"AAPL": 90.0}, start + timedelta(days=2))

    events = load_audit_events(engine.audit_event_path)
    assert [event.event_type for event in events] == [
        "entry_filled",
        "partial_exit_filled",
        "full_exit_filled",
    ]
    assert len({event.trade_id for event in events}) == 1
    assert engine.trades[0].parent_trade_id == engine.trades[1].trade_id
    assert engine.trades[1].decision_snapshot["confidence"] == 0.9
    assert engine.trades[1].regime_snapshot["trend_regime"] == "unknown"
