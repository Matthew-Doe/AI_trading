from __future__ import annotations

from datetime import datetime, timedelta, UTC
from trading_system.config import TradingConfig
from trading_system.backtest_execution import BacktestExecutionEngine, BacktestPosition
from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData, TradeDecision


def _symbol_data(symbol: str = "AAPL", price: float = 100.0, atr: float = 1.0) -> SymbolMarketData:
    return SymbolMarketData(
        symbol=symbol,
        market_cap=None,
        close=price,
        high_20d=price * 1.1,
        low_20d=price * 0.9,
        volume=1_000_000,
        indicators=IndicatorSnapshot(
            atr14=atr,
            rsi14=55.0,
            sma20=price,
            sma50=price,
            sma200=price,
            volatility20=0.2,
            avg_volume20=1_000_000,
        ),
        premarket=PremarketSnapshot(
            latest_price=price,
            gap_pct=0.0,
            volume=50_000,
            timestamp="2026-01-02T09:30:00+00:00",
        ),
        price_summary="",
    )


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
    assert engine.positions["AAPL"].confidence == 0.9
    assert engine.positions["AAPL"].allocation == 0.1
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
    assert engine.trades[0].exit_reason == "target_hit"
    assert engine.trades[0].confidence == 0.9
    assert engine.trades[0].allocation == 0.1
    assert engine.trades[0].return_pct == 0.0574
    assert engine.trades[0].risk_normalized_return == 0.0
    assert engine.cash == 10071.1


def test_confidence_sizing_experiment_halves_mid_confidence_size():
    config = TradingConfig(
        enable_confidence_sizing_experiment=True,
        max_single_trade_pct=0.10,
        cash_rich_available_cash_threshold=2.0,
        max_position_weight=0.50,
    )
    now = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.80, allocation=0.10)],
        {"AAPL": _symbol_data()},
        now,
    )

    assert engine.positions["AAPL"].qty == 5
    assert "confidence_multiplier=0.5000" in engine.positions["AAPL"].sizing_reason


def test_confidence_sizing_experiment_skips_below_floor_confidence():
    config = TradingConfig(
        enable_confidence_sizing_experiment=True,
        max_single_trade_pct=0.10,
        cash_rich_available_cash_threshold=2.0,
        max_position_weight=0.50,
    )
    now = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.69, allocation=0.10)],
        {"AAPL": _symbol_data()},
        now,
    )

    assert engine.positions == {}


def test_confidence_sizing_experiment_uses_full_size_for_high_confidence():
    config = TradingConfig(
        enable_confidence_sizing_experiment=True,
        max_single_trade_pct=0.10,
        high_confidence_trade_pct=0.20,
        cash_rich_available_cash_threshold=2.0,
        max_position_weight=0.50,
    )
    now = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.90, allocation=0.20)],
        {"AAPL": _symbol_data()},
        now,
    )

    assert engine.positions["AAPL"].qty == 20
    assert "cap_basis=high_confidence_cash" in engine.positions["AAPL"].sizing_reason

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


def test_backtest_execution_blocks_rebuy_after_loss_for_wash_sale_cooldown():
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=31,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=95.0,
            )
        ],
        {"AAPL": 100.0},
        start,
    )

    next_day = start + timedelta(days=1)
    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=110.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 90.0},
        next_day,
    )

    assert "AAPL" not in engine.positions
    assert len(engine.tax_blocked_decisions) == 1
    assert engine.tax_blocked_decisions[0]["reason"] == "wash_sale_loss_cooldown"
    assert len(engine.tax_shadow_positions) == 1
    assert engine.cash == 9900.0

    after_cooldown = next_day + timedelta(days=31)
    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 91.0},
        after_cooldown,
    )

    assert "AAPL" in engine.positions


def test_tax_shadow_trade_closes_without_affecting_real_portfolio_state():
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=31,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=95.0,
            )
        ],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=110.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 90.0},
        start + timedelta(days=1),
    )
    cash_after_block = engine.cash
    equity_after_block = engine.equity

    engine.process_decisions([], {"AAPL": 112.0}, start + timedelta(days=2))

    assert engine.cash == cash_after_block
    assert engine.equity == equity_after_block
    assert engine.positions == {}
    assert len(engine.trades) == 1
    assert len(engine.tax_shadow_trades) == 1
    assert engine.tax_shadow_trades[0].exit_reason == "target_hit"

    analysis = engine.get_tax_shadow_analysis()

    assert analysis["blocked_entry_count"] == 1
    assert analysis["closed_shadow_count"] == 1
    assert analysis["open_shadow_count"] == 0
    assert analysis["win_rate"] == 1.0
    assert analysis["average_return_pct"] == 0.2444
    assert analysis["missed_pnl_estimate"] == 242.0
    assert analysis["by_symbol"][0]["symbol"] == "AAPL"


def test_backtest_tax_summary_estimates_wash_sale_rebuy_loss_deferral():
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=0,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=95.0,
            )
        ],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions([], {"AAPL": 90.0}, start + timedelta(days=1))
    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 91.0},
        start + timedelta(days=2),
    )

    tax = engine.get_tax_summary()

    assert tax["loss_sale_count"] == 1
    assert tax["wash_sale_candidate_count"] == 1
    assert tax["wash_sale_disallowed_loss_estimate"] == 100.0
    assert tax["taxable_realized_pnl_estimate"] == 0.0


def test_exit_counterfactuals_track_delayed_long_pl_after_stop_loss():
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=0,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="AAPL",
                action="long",
                confidence=0.9,
                allocation=0.1,
                target_price=120.0,
                invalidation_price=95.0,
            )
        ],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions([], {"AAPL": 90.0}, start + timedelta(days=1))
    engine.process_decisions([], {"AAPL": 95.0}, start + timedelta(days=2))
    engine.process_decisions([], {"AAPL": 105.0}, start + timedelta(days=3))
    engine.process_decisions([], {"AAPL": 108.0}, start + timedelta(days=4))

    analysis = engine.get_exit_counterfactuals_analysis()
    stop_loss = analysis["by_exit_reason"]["stop_loss"]

    assert stop_loss["hold_1"]["count"] == 1
    assert stop_loss["hold_1"]["average_delayed_pnl"] == -50.0
    assert stop_loss["hold_1"]["average_delta_vs_actual"] == 50.0
    assert stop_loss["hold_1"]["recovery_rate"] == 1.0
    assert stop_loss["hold_1"]["delayed_win_rate"] == 0.0
    assert stop_loss["hold_3"]["average_delayed_pnl"] == 80.0
    assert stop_loss["hold_5"]["incomplete_count"] == 1


def test_exit_counterfactuals_track_delayed_short_pl_and_group_by_exit_reason():
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=0,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)

    engine.process_decisions(
        [
            TradeDecision(
                symbol="TSLA",
                action="short",
                confidence=0.8,
                allocation=0.1,
                target_price=80.0,
                invalidation_price=105.0,
            )
        ],
        {"TSLA": 100.0},
        start,
    )
    engine.process_decisions([], {"TSLA": 106.0}, start + timedelta(days=1))
    engine.process_decisions([], {"TSLA": 102.0}, start + timedelta(days=2))

    analysis = engine.get_exit_counterfactuals_analysis()
    stop_loss = analysis["by_exit_reason"]["stop_loss"]

    assert stop_loss["hold_1"]["count"] == 1
    assert stop_loss["hold_1"]["average_delayed_pnl"] == -20.0
    assert stop_loss["hold_1"]["average_delta_vs_actual"] == 40.0

if __name__ == "__main__":
    test_backtest_execution_long_entry_and_exit()
    test_backtest_execution_short_stop_loss()
    print("Execution Engine tests passed!")
