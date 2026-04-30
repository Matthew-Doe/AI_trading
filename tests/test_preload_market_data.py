from datetime import UTC, datetime

from preload_market_data import preload_alpaca_daily_bars
from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.utils import read_json


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_preload_alpaca_daily_bars_writes_symbol_cache_files(tmp_path):
    config = TradingConfig(
        cache_dir=tmp_path / ".cache",
        market_data_cache_dir=tmp_path / "market_bars",
    )
    service = MarketDataService(config, DummyLogger())

    def fake_fetch(symbols, start, end, feed):
        assert symbols == ["AAPL", "MSFT"]
        assert start == datetime(2026, 1, 1, tzinfo=UTC)
        assert end == datetime(2026, 1, 3, tzinfo=UTC)
        assert feed == "iex"
        return {
            "AAPL": [
                {
                    "t": "2026-01-02T05:00:00Z",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 12345,
                }
            ],
            "MSFT": [
                {
                    "t": "2026-01-02T05:00:00Z",
                    "o": 200.0,
                    "h": 201.0,
                    "l": 199.0,
                    "c": 200.5,
                    "v": 67890,
                }
            ],
        }

    service.fetch_alpaca_daily_bars_bulk = fake_fetch  # type: ignore[method-assign]

    summary = preload_alpaca_daily_bars(
        service,
        ["AAPL", "MSFT"],
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 3, tzinfo=UTC),
        batch_size=100,
        feed="iex",
    )

    assert summary == {"requested_symbols": 2, "cached_symbols": 2, "missing_symbols": []}
    payload = read_json(service.daily_bar_cache_dir / "AAPL.json")
    assert payload["symbol"] == "AAPL"
    assert payload["bars"][0]["close"] == 100.5
