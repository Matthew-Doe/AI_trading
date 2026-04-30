from datetime import UTC

import pandas as pd

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.main import load_mock_universe
from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData
from trading_system.utils import dataclass_to_dict, write_json


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


def _symbol_with_metrics(**overrides):
    indicators = overrides.pop(
        "indicators",
        IndicatorSnapshot(
            atr14=3.0,
            rsi14=55.0,
            sma20=101.0,
            sma50=99.0,
            sma200=95.0,
            volatility20=0.30,
            avg_volume20=1_000_000,
        ),
    )
    raw_metrics = {
        "price_change_5d": 0.02,
        "price_change_20d": 0.08,
        "volume_ratio": 1.2,
        "range_position": 0.7,
        "premarket_gap_pct": 0.01,
        "recent_volatility": 0.30,
    }
    raw_metrics.update(overrides.pop("raw_metrics", {}))
    return SymbolMarketData(
        symbol=overrides.pop("symbol", "AAPL"),
        market_cap=1_000_000_000,
        close=overrides.pop("close", 100.0),
        high_20d=overrides.pop("high_20d", 110.0),
        low_20d=overrides.pop("low_20d", 90.0),
        volume=overrides.pop("volume", 1_200_000),
        indicators=indicators,
        premarket=overrides.pop(
            "premarket",
            PremarketSnapshot(
                latest_price=101.0,
                gap_pct=0.01,
                volume=50_000,
                timestamp="2026-04-24T08:30:00-04:00",
            ),
        ),
        price_summary="AAPL clean setup",
        raw_metrics=raw_metrics,
        **overrides,
    )


def test_data_quality_marks_extreme_discontinuity_untradeable(tmp_path):
    config = TradingConfig(cache_dir=tmp_path / ".cache", max_symbol_20d_return=0.50)
    service = MarketDataService(config, DummyLogger())
    symbol_data = _symbol_with_metrics(raw_metrics={"price_change_20d": 0.95})

    checked = service.apply_data_quality(symbol_data)

    assert checked.is_tradeable is False
    assert "extreme_20d_return" in checked.data_quality_flags


def test_data_quality_allows_clean_symbol(tmp_path):
    config = TradingConfig(cache_dir=tmp_path / ".cache")
    service = MarketDataService(config, DummyLogger())
    symbol_data = _symbol_with_metrics()

    checked = service.apply_data_quality(symbol_data)

    assert checked.is_tradeable is True
    assert checked.data_quality_flags == []


def test_cached_market_data_is_rechecked_for_quality(tmp_path):
    config = TradingConfig(cache_dir=tmp_path / ".cache", max_symbol_20d_return=0.50)
    service = MarketDataService(config, DummyLogger())
    cached_symbol = _symbol_with_metrics(raw_metrics={"price_change_20d": 0.95})
    write_json(
        service.symbol_cache_dir / "AAPL.json",
        {
            "saved_at": "2026-04-24T12:00:00Z",
            "payload": dataclass_to_dict(cached_symbol),
        },
    )
    service._fetch_daily_bars = lambda symbol: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[method-assign]

    checked = service.fetch_symbol_market_data("AAPL", cached_symbol.market_cap)

    assert checked.is_tradeable is False
    assert "extreme_20d_return" in checked.data_quality_flags


def test_alpha_vantage_daily_bars_parse_ohlcv_response(tmp_path):
    config = TradingConfig(
        cache_dir=tmp_path / ".cache",
        alpha_vantage_api_key="demo-key",
    )
    service = MarketDataService(config, DummyLogger())

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "Time Series (Daily)": {
                    "2026-04-22": {
                        "1. open": "101.00",
                        "2. high": "105.00",
                        "3. low": "100.00",
                        "4. close": "104.00",
                        "5. volume": "1234567",
                    },
                    "2026-04-21": {
                        "1. open": "100.00",
                        "2. high": "103.00",
                        "3. low": "99.00",
                        "4. close": "102.00",
                        "5. volume": "7654321",
                    },
                }
            }

    captured_params = {}

    def fake_get(url, params, timeout):
        captured_params.update(params)
        return FakeResponse()

    service.session.get = fake_get  # type: ignore[method-assign]

    daily = service._fetch_daily_bars_alpha_vantage(
        "AAPL",
        end=pd.Timestamp("2026-04-22", tz=UTC).to_pydatetime(),
    )

    assert captured_params == {
        "function": "TIME_SERIES_DAILY",
        "symbol": "AAPL",
        "outputsize": "full",
        "apikey": "demo-key",
    }
    assert list(daily.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(daily["Close"]) == [102.0, 104.0]
    assert daily.index[0] == pd.Timestamp("2026-04-21", tz=UTC)
