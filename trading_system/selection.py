from __future__ import annotations

from dataclasses import replace

from trading_system.models import SymbolMarketData
from trading_system.utils import clamp, safe_float


class CandidateSelector:
    def __init__(self, logger):
        self.logger = logger

    def score_symbol(self, symbol_data: SymbolMarketData) -> SymbolMarketData:
        metrics = symbol_data.raw_metrics
        indicators = symbol_data.indicators

        volatility_expansion = clamp(metrics["recent_volatility"] / 0.35, 0.0, 2.0)
        volume_spike = clamp((metrics["volume_ratio"] - 1.0) / 2.0, 0.0, 2.0)
        breakout_setup = self._breakout_score(symbol_data)
        gap_potential = clamp(abs(metrics["premarket_gap_pct"]) / 0.03, 0.0, 2.0)
        trend_quality = self._trend_quality(symbol_data)
        atr_efficiency = clamp(indicators.atr14 / max(symbol_data.close, 1e-9) / 0.03, 0.0, 2.0)

        score_breakdown = {
            "volatility_expansion": volatility_expansion * 0.25,
            "volume_spike": volume_spike * 0.20,
            "breakout_setup": breakout_setup * 0.25,
            "gap_potential": gap_potential * 0.15,
            "trend_quality": trend_quality * 0.10,
            "atr_efficiency": atr_efficiency * 0.05,
        }
        total_score = sum(score_breakdown.values())
        score_breakdown["total"] = total_score

        return replace(symbol_data, score_breakdown=score_breakdown)

    def select(self, universe: list[SymbolMarketData], candidate_count: int) -> list[SymbolMarketData]:
        scored = [self.score_symbol(item) for item in universe]
        ranked = sorted(scored, key=lambda item: item.score_breakdown["total"], reverse=True)
        selection = ranked[:candidate_count]
        self.logger.info(
            "Selected symbols: %s",
            ", ".join(
                f"{item.symbol}({item.score_breakdown['total']:.2f})" for item in selection
            ),
        )
        return selection

    def _breakout_score(self, symbol_data: SymbolMarketData) -> float:
        close = symbol_data.close
        high_20d = symbol_data.high_20d
        low_20d = symbol_data.low_20d
        if high_20d <= low_20d:
            return 0.0
        distance_to_high = abs(high_20d - close) / close
        distance_to_low = abs(close - low_20d) / close
        return clamp((0.08 - min(distance_to_high, distance_to_low)) / 0.08, 0.0, 1.5)

    def _trend_quality(self, symbol_data: SymbolMarketData) -> float:
        indicators = symbol_data.indicators
        close = symbol_data.close
        bullish_stack = close > indicators.sma20 > indicators.sma50
        bearish_stack = close < indicators.sma20 < indicators.sma50
        long_term_alignment = close > indicators.sma200 or close < indicators.sma200
        score = 0.0
        if bullish_stack or bearish_stack:
            score += 1.0
        if long_term_alignment:
            score += 0.5
        score += clamp(abs(indicators.rsi14 - 50.0) / 25.0, 0.0, 0.5)
        return safe_float(score)

