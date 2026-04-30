from __future__ import annotations

from datetime import UTC, datetime

from backtest_engine import build_backtest_report
from backtest_engine import write_live_paper_readiness_if_accepted
from backtest_engine import build_confidence_analysis
from trading_system.config import TradingConfig
from trading_system.backtest_execution import BacktestExecutionEngine, BacktestTradeRecord


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
    assert "confidence_analysis" in report
    assert "tax_shadow_analysis" in report
    assert "exit_counterfactuals" in report


def test_build_backtest_report_includes_bias_controls_and_acceptance_warnings(tmp_path):
    config = TradingConfig(
        run_dir=tmp_path / "runs",
        log_dir=tmp_path / "logs",
        backtest_bias_safe_mode=True,
        backtest_universe_mode="point_in_time",
        backtest_calibration_mode="walk_forward",
        backtest_entry_timing_mode="previous_close_decision_next_open_fill",
        backtest_intraday_exit_mode="daily_high_low_conservative",
        backtest_friction_model="realistic",
    )
    execution = BacktestExecutionEngine(initial_cash=100000.0, config=config)
    execution.friction_summary["total_friction_cost"] = 12.5
    execution.missing_intraday_bar_count = 3

    report = build_backtest_report(
        config=config,
        execution=execution,
        daily_stats=[],
        initial_cash=100000.0,
        start_date="2026-03-01",
        end_date="2026-04-20",
        status="completed",
        run_at=datetime(2026, 4, 26, tzinfo=UTC),
    )

    controls = report["bias_controls"]
    assert controls["bias_safe_mode"] is True
    assert controls["point_in_time_universe_enabled"] is True
    assert controls["calibration_mode"] == "walk_forward"
    assert controls["entry_timing_mode"] == "previous_close_decision_next_open_fill"
    assert controls["intraday_exit_mode"] == "daily_high_low_conservative"
    assert controls["friction_model"] == "realistic"
    assert controls["total_friction_cost"] == 12.5
    assert "missing_point_in_time_universe_snapshot" in controls["acceptance_warnings"]
    assert controls["acceptance_grade"] is False


def test_build_backtest_report_references_audit_event_file(tmp_path):
    config = TradingConfig(run_dir=tmp_path / "runs", log_dir=tmp_path / "logs")
    execution = BacktestExecutionEngine(initial_cash=100000.0)
    execution.audit_event_path = tmp_path / "trade_audit_events.jsonl"
    execution.audit_event_count = 2

    report = build_backtest_report(
        config=config,
        execution=execution,
        daily_stats=[],
        initial_cash=100000.0,
        start_date="2026-03-01",
        end_date="2026-04-20",
        status="completed",
        run_at=datetime(2026, 4, 26, tzinfo=UTC),
    )

    assert report["audit_event_file"] == str(execution.audit_event_path)
    assert report["audit_event_count"] == 2


def test_live_paper_readiness_file_written_only_for_accepted_holdout(tmp_path):
    config = TradingConfig(live_paper_readiness_path=str(tmp_path / "readiness.json"))
    accepted = {
        "metadata": {"strategy": "hold_tax_partial", "window_label": "holdout"},
        "bias_controls": {"acceptance_grade": True, "acceptance_warnings": []},
    }
    rejected = {
        "metadata": {"strategy": "hold_tax_partial", "window_label": "train"},
        "bias_controls": {"acceptance_grade": True, "acceptance_warnings": []},
    }

    assert write_live_paper_readiness_if_accepted(config=config, report=rejected) is False
    assert not (tmp_path / "readiness.json").exists()
    assert write_live_paper_readiness_if_accepted(config=config, report=accepted) is True
    assert (tmp_path / "readiness.json").exists()


def test_confidence_analysis_handles_empty_trades():
    analysis = build_confidence_analysis([])

    assert analysis["trade_count"] == 0
    assert analysis["buckets"]["0.50-0.60"]["trade_count"] == 0
    assert analysis["correlations"]["confidence_vs_return_pct"] == 0.0


def test_confidence_analysis_buckets_profit_factor_median_and_correlations():
    trades = [
        BacktestTradeRecord(
            symbol="AAA",
            side="long",
            entry_time=datetime(2026, 1, 2, tzinfo=UTC),
            entry_price=100.0,
            exit_time=datetime(2026, 1, 3, tzinfo=UTC),
            exit_price=110.0,
            exit_reason="target_hit",
            qty=10,
            net_pnl=100.0,
            confidence=0.55,
            allocation=0.10,
            return_pct=0.10,
            risk_normalized_return=2.0,
            mfe_pct=0.12,
            mae_pct=-0.01,
        ),
        BacktestTradeRecord(
            symbol="BBB",
            side="long",
            entry_time=datetime(2026, 1, 2, tzinfo=UTC),
            entry_price=100.0,
            exit_time=datetime(2026, 1, 3, tzinfo=UTC),
            exit_price=95.0,
            exit_reason="stop_loss",
            qty=10,
            net_pnl=-50.0,
            confidence=0.65,
            allocation=0.10,
            return_pct=-0.05,
            risk_normalized_return=-1.0,
            mfe_pct=0.02,
            mae_pct=-0.06,
        ),
        BacktestTradeRecord(
            symbol="CCC",
            side="short",
            entry_time=datetime(2026, 1, 2, tzinfo=UTC),
            entry_price=100.0,
            exit_time=datetime(2026, 1, 3, tzinfo=UTC),
            exit_price=97.0,
            exit_reason="thesis_failed",
            qty=10,
            net_pnl=30.0,
            confidence=0.66,
            allocation=0.05,
            return_pct=0.03,
            risk_normalized_return=0.6,
            mfe_pct=0.04,
            mae_pct=-0.02,
        ),
    ]

    analysis = build_confidence_analysis(trades)

    assert analysis["trade_count"] == 3
    assert analysis["average_return_pct"] == 0.0267
    assert analysis["median_return_pct"] == 0.03
    assert analysis["profit_factor"] == 2.6
    assert analysis["buckets"]["0.50-0.60"]["trade_count"] == 1
    assert analysis["buckets"]["0.50-0.60"]["win_rate"] == 1.0
    assert analysis["buckets"]["0.60-0.70"]["trade_count"] == 2
    assert analysis["buckets"]["0.60-0.70"]["win_rate"] == 0.5
    assert analysis["buckets"]["0.60-0.70"]["median_return_pct"] == -0.01
    assert analysis["buckets"]["0.60-0.70"]["profit_factor"] == 0.6
    assert analysis["buckets"]["0.60-0.70"]["stop_loss_rate"] == 0.5
    assert analysis["correlations"]["confidence_vs_return_pct"] < 0
    assert analysis["correlations"]["confidence_vs_win_loss"] < 0


def test_confidence_analysis_zero_variance_correlation_is_zero():
    trades = [
        BacktestTradeRecord(
            symbol="AAA",
            side="long",
            entry_time=datetime(2026, 1, 2, tzinfo=UTC),
            entry_price=100.0,
            qty=1,
            net_pnl=10.0,
            confidence=0.70,
            return_pct=0.10,
        ),
        BacktestTradeRecord(
            symbol="BBB",
            side="long",
            entry_time=datetime(2026, 1, 2, tzinfo=UTC),
            entry_price=100.0,
            qty=1,
            net_pnl=-10.0,
            confidence=0.70,
            return_pct=-0.10,
        ),
    ]

    analysis = build_confidence_analysis(trades)

    assert analysis["correlations"]["confidence_vs_return_pct"] == 0.0
