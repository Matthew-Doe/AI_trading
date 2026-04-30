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


def test_bias_safe_previous_close_entry_ignores_open_print():
    config = TradingConfig(
        backtest_entry_timing_mode="previous_close_decision_next_open_fill",
        max_single_trade_pct=0.10,
        cash_rich_available_cash_threshold=2.0,
        max_position_weight=0.50,
    )
    symbol_data = _symbol_data(price=100.0)
    symbol_data.premarket.latest_price = 110.0
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.8, allocation=0.1)],
        {"AAPL": symbol_data},
        datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert engine.positions["AAPL"].entry_price == 100.0


def test_open_plus_delay_fill_uses_delayed_price_not_decision_open():
    config = TradingConfig(
        backtest_entry_timing_mode="open_plus_delay_fill",
        max_single_trade_pct=0.10,
        cash_rich_available_cash_threshold=2.0,
        max_position_weight=0.50,
    )
    symbol_data = _symbol_data(price=100.0)
    symbol_data.premarket.latest_price = 101.0
    symbol_data.raw_metrics["delayed_fill_price"] = 103.0
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.8, allocation=0.1)],
        {"AAPL": symbol_data},
        datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert engine.positions["AAPL"].entry_price == 103.0


def test_conservative_daily_high_low_exit_chooses_stop_when_stop_and_target_touched():
    config = TradingConfig(backtest_intraday_exit_mode="daily_high_low_conservative")
    start = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)
    engine.positions["AAPL"] = BacktestPosition(
        symbol="AAPL",
        qty=10,
        entry_price=100.0,
        entry_time=start,
        side="long",
        stop_price=95.0,
        take_profit_price=110.0,
    )
    engine.cash = 9000.0
    snapshot = _symbol_data(price=105.0)
    snapshot.raw_metrics["daily_high"] = 112.0
    snapshot.raw_metrics["daily_low"] = 94.0

    engine.process_decisions([], {"AAPL": snapshot}, start + timedelta(days=1))

    assert "AAPL" not in engine.positions
    assert engine.trades[0].exit_reason == "stop_loss"
    assert engine.trades[0].exit_price == 95.0
    assert engine.trades[0].intraday_data_missing is True


def test_realistic_friction_caps_quantity_to_liquidity_and_records_cost():
    config = TradingConfig(
        backtest_friction_model="realistic",
        max_single_trade_pct=0.50,
        cash_rich_trade_pct=0.50,
        cash_rich_available_cash_threshold=0.0,
        max_position_weight=1.0,
    )
    snapshot = _symbol_data(price=10.0)
    snapshot.volume = 100
    snapshot.raw_metrics["spread_pct"] = 0.02
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.5)],
        {"AAPL": snapshot},
        datetime(2026, 1, 2, tzinfo=UTC),
    )

    position = engine.positions["AAPL"]
    assert position.qty == 10
    assert position.entry_price > 10.0
    assert "partial_fill=true" in position.sizing_reason
    assert engine.friction_summary["partial_fill_count"] == 1
    assert engine.friction_summary["total_friction_cost"] > 0


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


def test_conditional_hold_extension_defers_thesis_failed_exit():
    config = TradingConfig(
        enable_conditional_hold_extension=True,
        conditional_hold_extension_observations=2,
        conditional_hold_max_adverse_pct=0.05,
        backtest_min_thesis_days=2,
        backtest_max_hold_days=10,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.1)],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions([], {"AAPL": 98.0}, start + timedelta(days=2))
    engine.process_decisions([], {"AAPL": 97.0}, start + timedelta(days=3))

    assert "AAPL" in engine.positions
    assert engine.positions["AAPL"].thesis_failure_deferrals == 2
    assert [log["action"] for log in engine.exit_adjustment_logs] == [
        "defer_thesis_failure_exit",
        "defer_thesis_failure_exit",
    ]

    engine.process_decisions([], {"AAPL": 96.5}, start + timedelta(days=4))

    assert "AAPL" not in engine.positions
    assert engine.trades[0].exit_reason == "thesis_failed"


def test_conditional_hold_extension_exits_when_adverse_move_is_too_large():
    config = TradingConfig(
        enable_conditional_hold_extension=True,
        conditional_hold_extension_observations=3,
        conditional_hold_max_adverse_pct=0.03,
        backtest_min_thesis_days=2,
        backtest_max_hold_days=10,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

    engine.process_decisions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.9, allocation=0.1)],
        {"AAPL": 100.0},
        start,
    )
    engine.process_decisions([], {"AAPL": 96.0}, start + timedelta(days=2))

    assert "AAPL" not in engine.positions
    assert engine.trades[0].exit_reason == "thesis_failed"
    assert engine.exit_adjustment_logs == []


def test_partial_profit_taking_closes_fraction_and_keeps_residual_position():
    config = TradingConfig(
        enable_partial_profit_taking=True,
        partial_profit_take_fraction=0.50,
        partial_profit_trailing_stop_pct=0.10,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

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

    assert "AAPL" in engine.positions
    assert engine.positions["AAPL"].qty == 5
    assert engine.positions["AAPL"].partial_profit_taken is True
    assert engine.positions["AAPL"].stop_price == 100.8
    assert len(engine.trades) == 1
    assert engine.trades[0].exit_reason == "partial_target_hit"
    assert engine.trades[0].qty == 5
    assert engine.trades[0].net_pnl == 60.0
    assert engine.cash == 9560.0


def test_partial_profit_taking_residual_closes_on_later_stop():
    config = TradingConfig(
        enable_partial_profit_taking=True,
        partial_profit_take_fraction=0.50,
        partial_profit_trailing_stop_pct=0.10,
    )
    start = datetime(2026, 1, 2, tzinfo=UTC)
    engine = BacktestExecutionEngine(initial_cash=10000.0, slippage_pct=0.0, config=config)

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
    engine.process_decisions([], {"AAPL": 100.0}, start + timedelta(days=2))

    assert "AAPL" not in engine.positions
    assert [trade.exit_reason for trade in engine.trades] == [
        "partial_target_hit",
        "breakeven_stop",
    ]
    assert engine.trades[1].qty == 5


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


def test_tax_adjusted_reentry_allows_high_ev_downsized_trade():
    config = TradingConfig(
        enable_tax_adjusted_ev_reentry=True,
        tax_reentry_min_confidence=0.90,
        tax_reentry_min_expected_value_pct=2.0,
        tax_reentry_size_multiplier=0.50,
    )
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=31,
        config=config,
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
                confidence=0.95,
                allocation=0.1,
                expected_value_pct=3.5,
                target_price=120.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 90.0},
        start + timedelta(days=1),
    )

    assert "AAPL" in engine.positions
    assert engine.positions["AAPL"].qty == 5
    assert engine.positions["AAPL"].allocation == 0.05
    assert "tax_adjusted_reentry=true" in engine.positions["AAPL"].sizing_reason
    assert engine.tax_blocked_decisions[-1]["reason"] == "tax_adjusted_ev_reentry_allowed"
    assert engine.tax_blocked_decisions[-1]["adjusted_allocation"] == 0.05
    assert len(engine.tax_shadow_positions) == 0


def test_tax_adjusted_reentry_still_blocks_low_ev_trade():
    config = TradingConfig(
        enable_tax_adjusted_ev_reentry=True,
        tax_reentry_min_confidence=0.90,
        tax_reentry_min_expected_value_pct=2.0,
        tax_reentry_size_multiplier=0.50,
    )
    engine = BacktestExecutionEngine(
        initial_cash=10000.0,
        slippage_pct=0.0,
        wash_sale_cooldown_days=31,
        config=config,
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
                confidence=0.95,
                allocation=0.1,
                expected_value_pct=1.5,
                target_price=120.0,
                invalidation_price=80.0,
            )
        ],
        {"AAPL": 90.0},
        start + timedelta(days=1),
    )

    assert "AAPL" not in engine.positions
    assert engine.tax_blocked_decisions[-1]["reason"] == "wash_sale_loss_cooldown"
    assert len(engine.tax_shadow_positions) == 1


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
