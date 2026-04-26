from __future__ import annotations

from datetime import datetime, UTC
from trading_system.backtest_execution import BacktestExecutionEngine, BacktestPosition
from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData, TradeDecision

def test_backtest_execution_long_entry_and_exit():
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.01) # High slippage for testing
    now = datetime.now(UTC)
    
    # 1. Test Entry
    decisions = [
        TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.1, target_price=160.0, invalidation_price=140.0)
    ]
    prices = {"AAPL": 150.0}
    
    engine.process_decisions(decisions, prices, now)
    
    # AAPL fill price = 150 * 1.01 = 151.5
    # Allocation 10% of 10000 = 1000
    # Qty = 1000 / 151.5 = 6
    # Cash used = 6 * 151.5 = 909
    
    assert "AAPL" in engine.positions
    assert engine.positions["AAPL"].qty == 6
    assert engine.positions["AAPL"].entry_price == 151.5
    assert engine.cash == 10000.0 - 909.0
    
    # 2. Test Equity Update
    engine.update_equity({"AAPL": 160.0})
    # Equity = (10000 - 909) + (6 * 160) = 9091 + 960 = 10051
    assert engine.equity == 10051.0
    
    # 3. Test Take Profit Exit
    # Process decisions again with higher price to trigger TP
    engine.process_decisions([], {"AAPL": 165.0}, now)
    # TP triggered at 165
    # Exit Fill = 165 * 0.99 = 163.35
    # Proceeds = 6 * 163.35 = 980.1
    # New Cash = 9091 + 980.1 = 10071.1
    
    assert "AAPL" not in engine.positions
    assert len(engine.trades) == 1
    assert engine.trades[0].exit_reason == "take_profit"
    assert engine.cash == 10071.1

def test_backtest_execution_short_stop_loss():
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.01)
    now = datetime.now(UTC)
    
    decisions = [
        TradeDecision(symbol="TSLA", action="short", confidence=0.8, allocation=0.2, target_price=100.0, invalidation_price=250.0)
    ]
    prices = {"TSLA": 200.0}
    
    engine.process_decisions(decisions, prices, now)
    
    # Short Fill = 200 * 0.99 = 198.0
    # Alloc = 0.2 capped at 0.15. 15% of 10000 = 1500.
    # Qty = 1500 / 198 = 7.
    # Cost = 7 * 198 = 1386. Cash = 10000 - 1386 = 8614.
    assert engine.positions["TSLA"].qty == 7
    
    # Trigger Stop Loss (Price goes up to 260)
    engine.process_decisions([], {"TSLA": 260.0}, now)
    # Stop triggered at 260. Exit Fill = 260 * 1.01 = 262.6
    # Gross PnL = (198 - 262.6) * 7 = -64.6 * 7 = -452.2
    # Proceeds = (7 * 198) + (-452.2) = 1386 - 452.2 = 933.8
    # Final Cash = 8614 + 933.8 = 9547.8
    
    assert engine.trades[0].exit_reason == "stop_loss"
    assert round(engine.cash, 1) == 9547.8
    assert round(engine.equity, 1) == 9547.8


def test_backtest_execution_uses_simulated_open_for_new_entries():
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.01)
    now = datetime(2026, 3, 2, tzinfo=UTC)
    symbol_data = SymbolMarketData(
        symbol="AAPL",
        market_cap=None,
        close=150.0,
        high_20d=160.0,
        low_20d=140.0,
        volume=1_000_000,
        indicators=IndicatorSnapshot(
            atr14=2.0,
            rsi14=55.0,
            sma20=150.0,
            sma50=145.0,
            sma200=140.0,
            volatility20=0.2,
            avg_volume20=1_000_000,
        ),
        premarket=PremarketSnapshot(
            latest_price=152.0,
            gap_pct=0.01,
            volume=50_000,
            timestamp="2026-03-02T09:30:00+00:00",
        ),
        price_summary="",
    )

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.1)],
        {"AAPL": symbol_data},
        now,
    )

    assert engine.positions["AAPL"].entry_price == 153.52

if __name__ == "__main__":
    test_backtest_execution_long_entry_and_exit()
    test_backtest_execution_short_stop_loss()
    print("Execution Engine tests passed!")
