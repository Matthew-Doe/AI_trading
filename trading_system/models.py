from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class IndicatorSnapshot:
    atr14: float
    rsi14: float
    sma20: float
    sma50: float
    sma200: float
    volatility20: float
    avg_volume20: float


@dataclass(slots=True)
class PremarketSnapshot:
    latest_price: float | None
    gap_pct: float | None
    volume: float | None
    timestamp: str | None


@dataclass(slots=True)
class SymbolMarketData:
    symbol: str
    market_cap: float | None
    close: float
    high_20d: float
    low_20d: float
    volume: float
    indicators: IndicatorSnapshot
    premarket: PremarketSnapshot
    price_summary: str
    news_headlines: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)
    data_quality_flags: list[str] = field(default_factory=list)
    is_tradeable: bool = True

    def to_prompt_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DebateResult:
    symbol: str
    position: str
    confidence: float
    arguments: list[str]
    risks: list[str]
    key_levels: dict[str, float | None]
    raw_response: str = ""


@dataclass(slots=True)
class SymbolDebate:
    symbol: str
    market_data: SymbolMarketData
    bull_case: DebateResult
    bear_case: DebateResult


@dataclass(slots=True)
class TradeDecision:
    symbol: str
    action: str
    confidence: float
    allocation: float
    expected_move_pct: float | None = None
    target_price: float | None = None
    invalidation_price: float | None = None
    time_horizon: str | None = None
    catalyst: str | None = None
    reward_risk_ratio: float | None = None


@dataclass(slots=True)
class OrderPlan:
    symbol: str
    side: str
    qty: int
    notional: float
    confidence: float
    allocation: float
    reason: str
    max_trade_pct: float = 0.0
    telegram_approval_required: bool = False
    telegram_approval_granted: bool = False
    entry_limit_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    risk_notional: float = 0.0
    order_style: str = "market"


@dataclass(slots=True)
class HeldPositionSignal:
    symbol: str
    current_side: str
    signal: str
    confidence: float
    current_qty: int
    target_qty: int
    delta_qty: int
    reason: str
    max_trade_pct: float = 0.0


@dataclass(slots=True)
class PendingOrderReview:
    symbol: str
    order_id: str | None
    side: str
    status: str
    submitted_at: str | None
    reference_close: float | None
    extended_hours_price: float | None
    price_change_pct: float | None
    action: str
    reason: str


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    started_at: datetime
    universe: list[SymbolMarketData]
    selected_symbols: list[SymbolMarketData]
    debates: list[SymbolDebate]
    decisions: list[TradeDecision]
    order_plans: list[OrderPlan]
