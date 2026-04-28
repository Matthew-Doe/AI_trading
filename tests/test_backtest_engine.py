from __future__ import annotations

from datetime import UTC, datetime

from backtest_engine import build_backtest_report
from trading_system.config import TradingConfig
from trading_system.backtest_execution import BacktestExecutionEngine


def test_build_backtest_report_documents_assumptions_and_limitations(tmp_path):
    config = TradingConfig(run_dir=tmp_path / "runs", log_dir=tmp_path / "logs")
    execution = BacktestExecutionEngine(initial_cash=100000.0)

    report = build_backtest_report(
        config=config,
        execution=execution,
        daily_stats=[],
        initial_cash=100000.0,
        start_date="2026-03-01",
        end_date="2026-04-20",
        status="in_progress",
        run_at=datetime(2026, 4, 26, tzinfo=UTC),
    )

    metadata = report["metadata"]
    assert metadata["universe_source"] == "current_companiesmarketcap_snapshot"
    assert metadata["entry_price_rule"] == "simulated_open_or_close_with_slippage"
    assert metadata["exit_price_rule"] == "daily_close_stop_target_or_staged_thesis_exit"
    assert metadata["sizing_rule"] == "live_style_risk_allocation_cash_and_position_caps"
    assert metadata["staged_exit"]["max_hold_days"] == 7
    assert metadata["tax_rule"] == "wash_sale_loss_cooldown_and_same_symbol_rebuy_estimate"
    assert metadata["wash_sale_cooldown_days"] == 31
    assert "not point-in-time" in metadata["known_limitations"][0]
    assert report["tax"]["wash_sale_cooldown_days"] == 31
