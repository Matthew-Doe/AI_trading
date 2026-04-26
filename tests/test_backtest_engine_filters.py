from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData, TradeDecision

from backtest_engine import apply_spy_regime_filter


def _symbol_data(symbol: str, *, close: float, sma20: float, sma50: float, change_5d: float) -> SymbolMarketData:
    return SymbolMarketData(
        symbol=symbol,
        market_cap=None,
        close=close,
        high_20d=close * 1.1,
        low_20d=close * 0.9,
        volume=1_000_000,
        indicators=IndicatorSnapshot(
            atr14=2.0,
            rsi14=55.0,
            sma20=sma20,
            sma50=sma50,
            sma200=sma50 * 0.9,
            volatility20=0.2,
            avg_volume20=1_000_000,
        ),
        premarket=PremarketSnapshot(
            latest_price=close,
            gap_pct=0.0,
            volume=100_000,
            timestamp=None,
        ),
        price_summary="",
        raw_metrics={"price_change_5d": change_5d},
    )


def test_spy_regime_filter_skips_new_longs_when_benchmark_is_bearish():
    decisions = [
        TradeDecision(symbol="AAPL", action="long", confidence=0.8, allocation=0.2),
        TradeDecision(symbol="MSFT", action="skip", confidence=0.0, allocation=0.0),
    ]
    prices = {
        "SPY": _symbol_data("SPY", close=95.0, sma20=100.0, sma50=101.0, change_5d=-0.03),
    }

    filtered = apply_spy_regime_filter(decisions, prices, benchmark_symbol="SPY", enabled=True)

    assert filtered[0].action == "skip"
    assert filtered[0].allocation == 0.0
    assert filtered[1].action == "skip"


def test_spy_regime_filter_allows_longs_when_benchmark_is_constructive():
    decisions = [
        TradeDecision(symbol="AAPL", action="long", confidence=0.8, allocation=0.2),
    ]
    prices = {
        "SPY": _symbol_data("SPY", close=105.0, sma20=100.0, sma50=99.0, change_5d=0.01),
    }

    filtered = apply_spy_regime_filter(decisions, prices, benchmark_symbol="SPY", enabled=True)

    assert filtered[0].action == "long"
    assert filtered[0].allocation == 0.2
