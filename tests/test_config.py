import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import trading_system.config as config_module
from trading_system.config import TradingConfig, _parse_schedule_times
from trading_system.data import MarketDataService


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
