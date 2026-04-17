from types import SimpleNamespace

from trading_system.config import TradingConfig
from trading_system.portfolio_summary import (
    MarketCloseReporter,
    MarketCloseSummary,
    build_market_close_summary_message,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeTradingClient:
    def get_account(self):
        return SimpleNamespace(equity="102500", last_equity="100000", cash="40500")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AAPL"), SimpleNamespace(symbol="MSFT")]


class FakeMarketData:
    def fetch_close_to_close_return(self, symbol: str):
        assert symbol == "SPY"
        return 500.0, 505.0, 0.01


class FakeTelegram:
    def __init__(self):
        self.messages: list[str] = []

    def is_enabled(self) -> bool:
        return True

    def send_message(self, message: str) -> None:
        self.messages.append(message)


def test_build_market_close_summary_message_uses_relative_performance():
    summary = MarketCloseSummary(
        as_of="2026-04-16T16:05:00-04:00",
        equity=102500.0,
        last_equity=100000.0,
        cash=40500.0,
        position_count=2,
        benchmark_symbol="SPY",
        benchmark_previous_close=500.0,
        benchmark_close=505.0,
        portfolio_return=0.025,
        benchmark_return=0.01,
        relative_return=0.015,
    )

    message = build_market_close_summary_message(summary)

    assert "Portfolio: $100,000.00 -> $102,500.00 (2.50%)" in message
    assert "SPY: $500.00 -> $505.00 (1.00%)" in message
    assert "Relative: outperformed by 1.50%" in message


def test_market_close_reporter_sends_telegram_message():
    telegram = FakeTelegram()
    reporter = MarketCloseReporter(
        TradingConfig(benchmark_symbol="SPY"),
        DummyLogger(),
        market_data=FakeMarketData(),
        trading_client=FakeTradingClient(),
        telegram=telegram,
    )

    summary = reporter.send_summary()

    assert round(summary.portfolio_return, 4) == 0.025
    assert round(summary.relative_return, 4) == 0.015
    assert len(telegram.messages) == 1
    assert "Market close summary" in telegram.messages[0]
