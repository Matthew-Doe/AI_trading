from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.telegram import TelegramNotifier


@dataclass(slots=True)
class MarketCloseSummary:
    as_of: str
    equity: float
    last_equity: float
    cash: float
    position_count: int
    benchmark_symbol: str
    benchmark_previous_close: float
    benchmark_close: float
    portfolio_return: float
    benchmark_return: float
    relative_return: float


class MarketCloseReporter:
    def __init__(
        self,
        config: TradingConfig,
        logger,
        market_data: MarketDataService | None = None,
        trading_client: TradingClient | None = None,
        telegram: TelegramNotifier | None = None,
    ):
        self.config = config
        self.logger = logger
        self.market_data = market_data or MarketDataService(config, logger)
        self.trading_client = trading_client or TradingClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper=True,
            url_override=config.alpaca_paper_base_url,
        )
        self.telegram = telegram or TelegramNotifier(config, logger)

    def build_summary(self) -> MarketCloseSummary:
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()
        equity = float(account.equity)
        last_equity = float(getattr(account, "last_equity", 0.0) or 0.0)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        portfolio_return = ((equity - last_equity) / last_equity) if last_equity else 0.0
        benchmark_previous_close, benchmark_close, benchmark_return = (
            self.market_data.fetch_close_to_close_return(self.config.benchmark_symbol)
        )
        now = datetime.now(ZoneInfo(self.config.market_timezone))
        return MarketCloseSummary(
            as_of=now.isoformat(),
            equity=equity,
            last_equity=last_equity,
            cash=cash,
            position_count=len(positions),
            benchmark_symbol=self.config.benchmark_symbol,
            benchmark_previous_close=benchmark_previous_close,
            benchmark_close=benchmark_close,
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            relative_return=portfolio_return - benchmark_return,
        )

    def send_summary(self) -> MarketCloseSummary:
        if not self.telegram.is_enabled():
            raise ValueError("Telegram is not configured for market-close summaries.")
        summary = self.build_summary()
        self.telegram.send_message(build_market_close_summary_message(summary))
        return summary


def build_market_close_summary_message(summary: MarketCloseSummary) -> str:
    comparison = "matched"
    if summary.relative_return > 0:
        comparison = "outperformed"
    elif summary.relative_return < 0:
        comparison = "underperformed"

    return (
        f"Market close summary\n"
        f"As of: {summary.as_of}\n"
        f"Portfolio: ${summary.last_equity:,.2f} -> ${summary.equity:,.2f} "
        f"({summary.portfolio_return:.2%})\n"
        f"{summary.benchmark_symbol}: ${summary.benchmark_previous_close:,.2f} -> "
        f"${summary.benchmark_close:,.2f} ({summary.benchmark_return:.2%})\n"
        f"Relative: {comparison} by {abs(summary.relative_return):.2%}\n"
        f"Open positions: {summary.position_count}\n"
        f"Cash: ${summary.cash:,.2f}"
    )


def send_market_close_summary(config: TradingConfig, logger) -> MarketCloseSummary:
    reporter = MarketCloseReporter(config, logger)
    return reporter.send_summary()
