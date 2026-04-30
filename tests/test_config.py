import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import trading_system.config as config_module
from trading_system.config import TradingConfig, _parse_schedule_times
from trading_system.data import DataIngestionError, MarketDataService


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _fresh_trading_config() -> type[TradingConfig]:
    return importlib.reload(config_module).TradingConfig


def test_parse_schedule_times_multiple_entries():
    assert _parse_schedule_times("08:30,12:30,15:30", 8, 30) == (
        (8, 30),
        (12, 30),
        (15, 30),
    )


def test_trading_config_accepts_explicit_schedule_times(tmp_path):
    config = TradingConfig(
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "runs",
        cache_dir=tmp_path / ".cache",
        scheduled_times=((8, 30), (12, 30), (15, 30)),
    )

    assert config.scheduled_times == ((8, 30), (12, 30), (15, 30))


def test_trading_config_defaults_cover_cash_rich_cap_and_after_hours_summary(monkeypatch):
    monkeypatch.setenv("CASH_RICH_TRADE_PCT", "0.10")
    monkeypatch.delenv("MARKET_CLOSE_SUMMARY_HOUR_ET", raising=False)
    monkeypatch.delenv("MARKET_CLOSE_SUMMARY_MINUTE_ET", raising=False)

    config = _fresh_trading_config()()

    assert config.cash_rich_trade_pct == 0.10
    assert config.market_close_summary_hour == 20
    assert config.market_close_summary_minute == 0


def test_trading_config_defaults_confidence_actionable_move_pct(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_ACTIONABLE_MOVE_PCT", raising=False)

    config = _fresh_trading_config()()

    assert config.confidence_actionable_move_pct == 0.02


def test_trading_config_reads_alpha_vantage_settings(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "demo-key")
    monkeypatch.setenv("ALPHA_VANTAGE_CALLS_PER_MINUTE", "5")

    config = _fresh_trading_config()()

    assert config.alpha_vantage_api_key == "demo-key"
    assert config.alpha_vantage_calls_per_minute == 5


def test_behavior_experiment_flags_default_disabled(monkeypatch):
    for name in (
        "ENABLE_TAX_LOSS_SIZE_COOLDOWN_TIERS",
        "ENABLE_CONFIDENCE_SIZING_EXPERIMENT",
        "ENABLE_TAX_ADJUSTED_EV_REENTRY",
        "ENABLE_PROFIT_PROTECTION_BANDS",
        "ENABLE_PARTIAL_PROFIT_TAKING",
        "ENABLE_CONDITIONAL_HOLD_EXTENSION",
        "ENABLE_STRICT_THESIS_FAILURE_REASONS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _fresh_trading_config()()

    assert config.enable_tax_loss_size_cooldown_tiers is False
    assert config.enable_confidence_sizing_experiment is False
    assert config.confidence_full_size_threshold == 0.90
    assert config.confidence_mid_size_floor == 0.70
    assert config.confidence_mid_size_multiplier == 0.50
    assert config.confidence_low_size_multiplier == 0.00
    assert config.enable_tax_adjusted_ev_reentry is False
    assert config.tax_reentry_min_confidence == 0.90
    assert config.tax_reentry_min_expected_value_pct == 2.00
    assert config.tax_reentry_size_multiplier == 0.50
    assert config.enable_profit_protection_bands is False
    assert config.enable_partial_profit_taking is False
    assert config.partial_profit_take_fraction == 0.60
    assert config.partial_profit_trailing_stop_pct == 0.03
    assert config.enable_conditional_hold_extension is False
    assert config.conditional_hold_extension_observations == 3
    assert config.conditional_hold_max_adverse_pct == 0.04
    assert config.enable_strict_thesis_failure_reasons is False


def test_behavior_experiment_flags_read_from_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_TAX_LOSS_SIZE_COOLDOWN_TIERS", "true")
    monkeypatch.setenv("ENABLE_CONFIDENCE_SIZING_EXPERIMENT", "true")
    monkeypatch.setenv("CONFIDENCE_FULL_SIZE_THRESHOLD", "0.91")
    monkeypatch.setenv("CONFIDENCE_MID_SIZE_FLOOR", "0.72")
    monkeypatch.setenv("CONFIDENCE_MID_SIZE_MULTIPLIER", "0.40")
    monkeypatch.setenv("CONFIDENCE_LOW_SIZE_MULTIPLIER", "0.10")
    monkeypatch.setenv("ENABLE_TAX_ADJUSTED_EV_REENTRY", "true")
    monkeypatch.setenv("TAX_REENTRY_MIN_CONFIDENCE", "0.92")
    monkeypatch.setenv("TAX_REENTRY_MIN_EXPECTED_VALUE_PCT", "3.00")
    monkeypatch.setenv("TAX_REENTRY_SIZE_MULTIPLIER", "0.25")
    monkeypatch.setenv("ENABLE_PROFIT_PROTECTION_BANDS", "true")
    monkeypatch.setenv("ENABLE_PARTIAL_PROFIT_TAKING", "true")
    monkeypatch.setenv("PARTIAL_PROFIT_TAKE_FRACTION", "0.50")
    monkeypatch.setenv("PARTIAL_PROFIT_TRAILING_STOP_PCT", "0.02")
    monkeypatch.setenv("ENABLE_CONDITIONAL_HOLD_EXTENSION", "true")
    monkeypatch.setenv("CONDITIONAL_HOLD_EXTENSION_OBSERVATIONS", "2")
    monkeypatch.setenv("CONDITIONAL_HOLD_MAX_ADVERSE_PCT", "0.03")
    monkeypatch.setenv("ENABLE_STRICT_THESIS_FAILURE_REASONS", "true")

    config = _fresh_trading_config()()

    assert config.enable_tax_loss_size_cooldown_tiers is True
    assert config.enable_confidence_sizing_experiment is True
    assert config.confidence_full_size_threshold == 0.91
    assert config.confidence_mid_size_floor == 0.72
    assert config.confidence_mid_size_multiplier == 0.40
    assert config.confidence_low_size_multiplier == 0.10
    assert config.enable_tax_adjusted_ev_reentry is True
    assert config.tax_reentry_min_confidence == 0.92
    assert config.tax_reentry_min_expected_value_pct == 3.00
    assert config.tax_reentry_size_multiplier == 0.25
    assert config.enable_profit_protection_bands is True
    assert config.enable_partial_profit_taking is True
    assert config.partial_profit_take_fraction == 0.50
    assert config.partial_profit_trailing_stop_pct == 0.02
    assert config.enable_conditional_hold_extension is True
    assert config.conditional_hold_extension_observations == 2
    assert config.conditional_hold_max_adverse_pct == 0.03
    assert config.enable_strict_thesis_failure_reasons is True


def test_bias_safe_backtest_config_defaults(monkeypatch):
    for name in (
        "BACKTEST_BIAS_SAFE_MODE",
        "BACKTEST_UNIVERSE_MODE",
        "BACKTEST_UNIVERSE_SNAPSHOT_DIR",
        "ALLOW_CURRENT_UNIVERSE_FALLBACK",
        "BACKTEST_CALIBRATION_MODE",
        "BACKTEST_ENTRY_TIMING_MODE",
        "BACKTEST_ENTRY_DELAY_MINUTES",
        "BACKTEST_INTRADAY_EXIT_MODE",
        "BACKTEST_FRICTION_MODEL",
        "LIVE_PAPER_REQUIRE_BIAS_SAFE_ACCEPTANCE",
        "LIVE_PAPER_READINESS_PATH",
        "ENABLE_LIVE_PAPER_BACKTEST_STYLE",
        "LIVE_PAPER_STRATEGY",
        "REQUIRE_EMPTY_PAPER_ACCOUNT",
        "ALLOW_LIVE_LARGE_TRADE_APPROVAL",
        "LIVE_ENTRY_REVIEW_TIME_ET",
        "LIVE_EXIT_REVIEW_TIME_ET",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _fresh_trading_config()()

    assert config.backtest_bias_safe_mode is False
    assert config.backtest_universe_mode == "current"
    assert config.backtest_universe_snapshot_dir == "data/universe_snapshots"
    assert config.allow_current_universe_fallback is False
    assert config.backtest_calibration_mode == "walk_forward"
    assert config.backtest_entry_timing_mode == "previous_close_decision_next_open_fill"
    assert config.backtest_entry_delay_minutes == 15
    assert config.backtest_intraday_exit_mode == "daily_high_low_conservative"
    assert config.backtest_friction_model == "basic"
    assert config.live_paper_require_bias_safe_acceptance is True
    assert config.live_paper_readiness_path == "runs/live_paper_readiness.json"
    assert config.enable_live_paper_backtest_style is False
    assert config.live_paper_strategy == "hold_tax_partial"
    assert config.require_empty_paper_account is True
    assert config.allow_live_large_trade_approval is False
    assert config.live_entry_review_time_et == "09:45"
    assert config.live_exit_review_time_et == "15:45"


def test_bias_safe_backtest_config_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BACKTEST_BIAS_SAFE_MODE", "true")
    monkeypatch.setenv("BACKTEST_UNIVERSE_MODE", "point_in_time")
    monkeypatch.setenv("BACKTEST_UNIVERSE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("ALLOW_CURRENT_UNIVERSE_FALLBACK", "true")
    monkeypatch.setenv("BACKTEST_CALIBRATION_MODE", "off")
    monkeypatch.setenv("BACKTEST_ENTRY_TIMING_MODE", "open_plus_delay_fill")
    monkeypatch.setenv("BACKTEST_ENTRY_DELAY_MINUTES", "30")
    monkeypatch.setenv("BACKTEST_INTRADAY_EXIT_MODE", "intraday_bars")
    monkeypatch.setenv("BACKTEST_FRICTION_MODEL", "realistic")
    monkeypatch.setenv("LIVE_PAPER_REQUIRE_BIAS_SAFE_ACCEPTANCE", "false")
    monkeypatch.setenv("LIVE_PAPER_READINESS_PATH", str(tmp_path / "readiness.json"))
    monkeypatch.setenv("ENABLE_LIVE_PAPER_BACKTEST_STYLE", "true")
    monkeypatch.setenv("LIVE_PAPER_STRATEGY", "hold_tax_partial")
    monkeypatch.setenv("REQUIRE_EMPTY_PAPER_ACCOUNT", "false")
    monkeypatch.setenv("ALLOW_LIVE_LARGE_TRADE_APPROVAL", "true")
    monkeypatch.setenv("LIVE_ENTRY_REVIEW_TIME_ET", "10:05")
    monkeypatch.setenv("LIVE_EXIT_REVIEW_TIME_ET", "15:30")

    config = _fresh_trading_config()()

    assert config.backtest_bias_safe_mode is True
    assert config.backtest_universe_mode == "point_in_time"
    assert config.backtest_universe_snapshot_dir == str(tmp_path / "snapshots")
    assert config.allow_current_universe_fallback is True
    assert config.backtest_calibration_mode == "off"
    assert config.backtest_entry_timing_mode == "open_plus_delay_fill"
    assert config.backtest_entry_delay_minutes == 30
    assert config.backtest_intraday_exit_mode == "intraday_bars"
    assert config.backtest_friction_model == "realistic"
    assert config.live_paper_require_bias_safe_acceptance is False
    assert config.live_paper_readiness_path == str(tmp_path / "readiness.json")
    assert config.enable_live_paper_backtest_style is True
    assert config.live_paper_strategy == "hold_tax_partial"
    assert config.require_empty_paper_account is False
    assert config.allow_live_large_trade_approval is True
    assert config.live_entry_review_time_et == "10:05"
    assert config.live_exit_review_time_et == "15:30"


def test_fetch_forward_close_window_uses_next_three_trading_days():
    config = TradingConfig()
    service = MarketDataService.__new__(MarketDataService)
    service.config = config
    service.logger = DummyLogger()

    service._fetch_daily_bars = lambda symbol: pd.DataFrame(  # type: ignore[method-assign]
        {
            "Close": [100.0, 101.0, 102.0, 104.0, 103.0],
        },
        index=pd.to_datetime(
            [
                "2026-04-20T20:00:00Z",
                "2026-04-21T20:00:00Z",
                "2026-04-22T20:00:00Z",
                "2026-04-23T20:00:00Z",
                "2026-04-24T20:00:00Z",
            ],
            utc=True,
        ),
    )

    reference_close, forward_close, forward_as_of = service.fetch_forward_close_window(
        "AAPL",
        as_of=datetime(2026, 4, 21, 12, 30, tzinfo=ZoneInfo("America/New_York")),
        trading_days_ahead=3,
    )

    assert reference_close == 100.0
    assert forward_close == 104.0
    assert forward_as_of.startswith("2026-04-23")


def test_point_in_time_universe_snapshot_selects_latest_on_or_before_date(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "2026-01-01.json").write_text(
        '{"source":"test","symbols":[{"symbol":"AAPL","name":"Apple","market_cap":3000}]}',
        encoding="utf-8",
    )
    (snapshot_dir / "2026-02-01.json").write_text(
        '{"source":"test","symbols":[{"symbol":"MSFT","name":"Microsoft","market_cap":2900}]}',
        encoding="utf-8",
    )
    config = TradingConfig(
        backtest_universe_mode="point_in_time",
        backtest_universe_snapshot_dir=str(snapshot_dir),
        top_universe_size=10,
        index_proxy_symbols=(),
    )
    service = MarketDataService.__new__(MarketDataService)
    service.config = config
    service.logger = DummyLogger()

    companies = service._build_symbol_universe(as_of_date=datetime(2026, 1, 15))

    assert companies == [{"symbol": "AAPL", "name": "Apple", "market_cap": 3000}]
    assert service.last_universe_metadata["snapshot_date"] == "2026-01-01"
    assert service.last_universe_metadata["fallback_used"] is False


def test_point_in_time_universe_snapshot_does_not_truncate_snapshot_symbols(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    symbols = [
        {"symbol": f"SYM{index}", "name": f"Symbol {index}", "market_cap": None}
        for index in range(12)
    ]
    (snapshot_dir / "2025-10-01.json").write_text(
        __import__("json").dumps({"source": "test", "symbols": symbols}),
        encoding="utf-8",
    )
    config = TradingConfig(
        backtest_universe_mode="point_in_time",
        backtest_universe_snapshot_dir=str(snapshot_dir),
        top_universe_size=5,
        index_proxy_symbols=(),
    )
    service = MarketDataService.__new__(MarketDataService)
    service.config = config
    service.logger = DummyLogger()

    companies = service._build_symbol_universe(as_of_date=datetime(2025, 10, 15))

    assert len(companies) == 12
    assert companies[-1]["symbol"] == "SYM11"
    assert service.last_universe_metadata["symbol_count"] == 12


def test_point_in_time_universe_snapshot_fails_closed_without_snapshot(tmp_path):
    config = TradingConfig(
        backtest_universe_mode="point_in_time",
        backtest_universe_snapshot_dir=str(tmp_path / "missing"),
        allow_current_universe_fallback=False,
    )
    service = MarketDataService.__new__(MarketDataService)
    service.config = config
    service.logger = DummyLogger()

    try:
        service._build_symbol_universe(as_of_date=datetime(2026, 1, 15))
    except DataIngestionError as exc:
        assert "No point-in-time universe snapshot" in str(exc)
    else:
        raise AssertionError("missing snapshot should fail closed")


def test_bias_safe_historical_premarket_snapshot_does_not_use_full_day_volume():
    config = TradingConfig(backtest_bias_safe_mode=True)
    service = MarketDataService.__new__(MarketDataService)
    service.config = config
    service.logger = DummyLogger()
    service._fetch_daily_bars = lambda symbol, as_of_date=None: pd.DataFrame(  # type: ignore[method-assign]
        {
            "Open": [99.0, 102.0],
            "High": [101.0, 105.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 104.0],
            "Volume": [1_000_000, 9_000_000],
        },
        index=pd.to_datetime(["2026-01-01T21:00:00Z", "2026-01-02T21:00:00Z"], utc=True),
    )

    snapshot = service._fetch_premarket_snapshot(
        "AAPL",
        last_close=100.0,
        as_of_date=datetime(2026, 1, 2),
    )

    assert snapshot.latest_price == 102.0
    assert snapshot.volume is None
