from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.main import load_mock_universe


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_symbol_universe_appends_index_proxies_without_duplicates(tmp_path):
    config = TradingConfig(
        cache_dir=tmp_path / ".cache",
        index_proxy_symbols=("SPY", "QQQ", "AAPL"),
    )
    service = MarketDataService(config, DummyLogger())
    service.fetch_top_us_companies_by_market_cap = lambda limit=None: [  # type: ignore[method-assign]
        {"symbol": "AAPL", "name": "Apple", "market_cap": 1.0},
        {"symbol": "MSFT", "name": "Microsoft", "market_cap": 2.0},
    ]

    symbols = service._build_symbol_universe()

    assert [item["symbol"] for item in symbols] == ["AAPL", "MSFT", "SPY", "QQQ"]
    assert next(item for item in symbols if item["symbol"] == "SPY")["market_cap"] is None


def test_load_mock_universe_includes_index_proxies():
    symbols = {item.symbol for item in load_mock_universe()}

    assert {"SPY", "QQQ", "DIA", "IWM"}.issubset(symbols)


def test_symbol_universe_falls_back_to_cached_symbol_files(tmp_path):
    config = TradingConfig(
        cache_dir=tmp_path / ".cache",
        top_universe_size=2,
        index_proxy_symbols=("SPY",),
    )
    service = MarketDataService(config, DummyLogger())
    service.fetch_top_us_companies_by_market_cap = lambda limit=None: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("network unavailable")
    )

    for symbol, market_cap in (("MSFT", 3.0), ("AAPL", 2.0), ("TSLA", 1.0)):
        payload = {
            "saved_at": "2026-04-16T00:00:00Z",
            "payload": {
                "symbol": symbol,
                "market_cap": market_cap,
            },
        }
        (service.symbol_cache_dir / f"{symbol}.json").write_text(str(payload).replace("'", '"'), encoding="utf-8")

    symbols = service._build_symbol_universe()

    assert [item["symbol"] for item in symbols] == ["MSFT", "AAPL", "SPY"]
