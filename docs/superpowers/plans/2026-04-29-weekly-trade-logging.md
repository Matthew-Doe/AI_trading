# Weekly Trade Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build enough structured trade logging to support weekly strategy updates, regime adaptation, live-paper debugging, and backtest-to-live comparison.

**Architecture:** Add an append-only audit layer that records every material decision, order, fill, position update, exit adjustment, and post-trade outcome with a stable `trade_id`. Keep the current backtest report fields, but enrich them with decision snapshots, market/regime snapshots, order lifecycle data, and weekly summary analytics. The weekly review script should read both backtest reports and live paper audit logs and produce a compact markdown report that answers what changed, what worked, what failed, and which strategy parameters deserve review.

**Tech Stack:** Python dataclasses, JSON/JSONL files, existing `TradingConfig`, existing backtest/live execution engines, pytest.

---

## Operating Requirements

- Preserve backward compatibility with existing `backtest_report.json` files.
- Do not remove the current `all_trades`, `confidence_analysis`, `tax_shadow_analysis`, or `exit_counterfactuals` outputs.
- Make each closed trade auditable from signal to exit:
  - selected-symbol score and rank
  - market snapshot at entry decision
  - bull/bear debate summary and confidence
  - raw confidence, calibrated confidence, and cap reason
  - decision expected value fields
  - sizing inputs and binding constraint
  - intended order plan
  - actual or simulated fill
  - stop/target/partial/hold changes
  - final exit reason
  - MFE/MAE and counterfactual outcomes
- Add regime tags that can be grouped weekly:
  - broad market trend
  - volatility regime
  - gap regime
  - liquidity/volume regime
  - symbol momentum regime
  - sector or industry if available later
- Live paper logging must include broker order identifiers, order status transitions, partial fills, rejects, cancels, and final fill prices.
- Weekly reports should be useful for adapting the strategy, not just reporting performance.
- Keep logs local-first and deterministic. No database is required for this phase.

## File Structure

- Create `trading_system/trade_audit.py`: audit dataclasses, stable ID helpers, JSONL writer, and report parsing helpers.
- Modify `trading_system/models.py`: add optional `trade_id`, `decision_id`, and `audit_context` fields where needed.
- Modify `trading_system/backtest_execution.py`: attach `trade_id` to positions/trade records and emit audit events.
- Modify `backtest_engine.py`: build decision context maps, pass them into execution, and include a `trade_audit` section in reports.
- Modify `trading_system/execution.py`: emit live order lifecycle and broker fill events.
- Modify `trading_system/reporting.py`: include audit summaries in live run reports and write `trade_audit_events.jsonl`.
- Create `trading_system/weekly_review.py`: generate weekly markdown and JSON summaries from backtest reports or live audit logs.
- Create `scripts/weekly_trade_review.py`: CLI wrapper for weekly review generation.
- Modify `analyze_backtest_correlation.py`: read the new audit fields for richer group-by analysis.
- Test `tests/test_trade_audit.py`: audit serialization, stable IDs, JSONL append/read.
- Test `tests/test_backtest_audit_logging.py`: backtest trade lifecycle context and report compatibility.
- Test `tests/test_live_audit_logging.py`: live broker event logging without requiring network calls.
- Test `tests/test_weekly_review.py`: weekly summary metrics and regime breakdowns.

---

### Task 1: Add Canonical Trade Audit Models

**Files:**
- Create: `trading_system/trade_audit.py`
- Test: `tests/test_trade_audit.py`

- [ ] **Step 1: Write failing audit model tests**

Create `tests/test_trade_audit.py`:

```python
from __future__ import annotations

import json

from trading_system.trade_audit import (
    AuditEvent,
    DecisionSnapshot,
    MarketRegimeSnapshot,
    append_audit_event,
    build_decision_id,
    build_trade_id,
    load_audit_events,
)


def test_build_trade_id_is_stable_and_human_readable():
    first = build_trade_id(
        run_id="20260429T195150Z",
        symbol="AAPL",
        side="long",
        entry_time="2026-01-05T00:00:00+00:00",
        sequence=3,
    )
    second = build_trade_id(
        run_id="20260429T195150Z",
        symbol="aapl",
        side="LONG",
        entry_time="2026-01-05T00:00:00+00:00",
        sequence=3,
    )

    assert first == second
    assert first == "20260429T195150Z:2026-01-05:AAPL:long:3"


def test_build_decision_id_is_stable():
    decision_id = build_decision_id(
        run_id="20260429T195150Z",
        symbol="MSFT",
        decision_time="2026-01-05T00:00:00+00:00",
    )

    assert decision_id == "20260429T195150Z:2026-01-05:MSFT"


def test_audit_event_jsonl_round_trip(tmp_path):
    path = tmp_path / "trade_audit_events.jsonl"
    event = AuditEvent(
        event_id="event-1",
        run_id="run-1",
        trade_id="run-1:2026-01-05:AAPL:long:1",
        decision_id="run-1:2026-01-05:AAPL",
        event_type="entry_fill",
        occurred_at="2026-01-05T00:00:00+00:00",
        symbol="AAPL",
        payload={"fill_price": 100.25, "qty": 12},
    )

    append_audit_event(path, event)
    loaded = load_audit_events(path)

    assert loaded == [event]
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    assert raw["event_type"] == "entry_fill"
    assert raw["payload"]["qty"] == 12


def test_snapshot_dataclasses_are_json_serializable():
    decision = DecisionSnapshot(
        symbol="NVDA",
        action="long",
        confidence=0.91,
        raw_confidence=0.96,
        calibrated_confidence=0.91,
        allocation=0.12,
        estimated_win_probability=0.64,
        expected_upside_pct=0.08,
        expected_downside_pct=0.03,
        expected_value_pct=0.035,
        risk_reward=2.7,
        evidence_count=6,
        confidence_cap_reason="calibration_cap",
        target_price=120.0,
        invalidation_price=108.0,
        reward_risk_ratio=3.0,
        reason="momentum confirmed",
        bull_confidence=0.88,
        bear_confidence=0.42,
        bull_arguments=["trend up"],
        bear_arguments=["valuation risk"],
        bull_risks=["gap fade"],
        bear_risks=["short squeeze"],
    )
    regime = MarketRegimeSnapshot(
        market_trend="uptrend",
        volatility_regime="normal",
        gap_regime="gap_up",
        volume_regime="high_volume",
        symbol_momentum_regime="strong_up",
        tags=["sma20_above_sma50", "rsi_neutral"],
    )

    payload = {
        "decision": decision.to_dict(),
        "regime": regime.to_dict(),
    }

    assert payload["decision"]["raw_confidence"] == 0.96
    assert payload["regime"]["tags"] == ["sma20_above_sma50", "rsi_neutral"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_trade_audit.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'trading_system.trade_audit'`.

- [ ] **Step 3: Implement audit models**

Create `trading_system/trade_audit.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _date_part(value: str) -> str:
    return value[:10]


def build_trade_id(
    *,
    run_id: str,
    symbol: str,
    side: str,
    entry_time: str,
    sequence: int,
) -> str:
    return f"{run_id}:{_date_part(entry_time)}:{symbol.upper()}:{side.lower()}:{sequence}"


def build_decision_id(*, run_id: str, symbol: str, decision_time: str) -> str:
    return f"{run_id}:{_date_part(decision_time)}:{symbol.upper()}"


@dataclass(slots=True)
class DecisionSnapshot:
    symbol: str
    action: str
    confidence: float
    raw_confidence: float | None
    calibrated_confidence: float | None
    allocation: float
    estimated_win_probability: float | None
    expected_upside_pct: float | None
    expected_downside_pct: float | None
    expected_value_pct: float | None
    risk_reward: float | None
    evidence_count: int | None
    confidence_cap_reason: str | None
    target_price: float | None
    invalidation_price: float | None
    reward_risk_ratio: float | None
    reason: str | None
    bull_confidence: float | None = None
    bear_confidence: float | None = None
    bull_arguments: list[str] = field(default_factory=list)
    bear_arguments: list[str] = field(default_factory=list)
    bull_risks: list[str] = field(default_factory=list)
    bear_risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MarketRegimeSnapshot:
    market_trend: str
    volatility_regime: str
    gap_regime: str
    volume_regime: str
    symbol_momentum_regime: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditEvent:
    event_id: str
    run_id: str
    trade_id: str | None
    decision_id: str | None
    event_type: str
    occurred_at: str
    symbol: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_audit_event(path: Path, event: AuditEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True))
        handle.write("\n")


def load_audit_events(path: Path) -> list[AuditEvent]:
    if not path.exists():
        return []
    events: list[AuditEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        events.append(AuditEvent(**payload))
    return events
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_trade_audit.py -q
```

Commit:

```bash
rtk git add trading_system/trade_audit.py tests/test_trade_audit.py
rtk git commit -m "feat: add trade audit log primitives"
```

---

### Task 2: Add Stable Trade IDs To Backtest Positions And Trades

**Files:**
- Modify: `trading_system/backtest_execution.py`
- Test: `tests/test_backtest_execution.py`

- [ ] **Step 1: Add failing trade ID test**

Append to `tests/test_backtest_execution.py`:

```python
def test_backtest_trade_records_include_stable_trade_id(sample_symbol_data):
    engine = BacktestExecutionEngine(initial_cash=100000.0, config=None)
    decision = TradeDecision(
        symbol="AAPL",
        action="long",
        confidence=0.9,
        allocation=0.10,
    )
    now = datetime(2026, 1, 5, tzinfo=UTC)

    engine.run_id = "unit-run"
    engine.process_decisions([decision], {"AAPL": 100.0}, now)
    engine.process_decisions([], {"AAPL": 95.0}, now + timedelta(days=8))

    assert len(engine.trades) == 1
    trade = engine.trades[0]
    assert trade.trade_id == "unit-run:2026-01-05:AAPL:long:1"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_backtest_execution.py::test_backtest_trade_records_include_stable_trade_id -q
```

Expected: fail because `BacktestTradeRecord` has no `trade_id`.

- [ ] **Step 3: Add trade ID fields and sequence**

In `trading_system/backtest_execution.py`, import:

```python
from trading_system.trade_audit import build_trade_id
```

Add fields:

```python
@dataclass
class BacktestPosition:
    trade_id: str = ""
```

```python
@dataclass
class BacktestTradeRecord:
    trade_id: str = ""
```

In `BacktestExecutionEngine.__init__`, add:

```python
self.run_id = "backtest"
self._next_trade_sequence = 1
```

Before opening a position in `process_decisions`, create:

```python
trade_id = build_trade_id(
    run_id=self.run_id,
    symbol=dec.symbol,
    side=dec.action,
    entry_time=current_time.isoformat(),
    sequence=self._next_trade_sequence,
)
self._next_trade_sequence += 1
```

Set `trade_id=trade_id` on `BacktestPosition`.

Set `trade_id=pos.trade_id` on both full-close and partial-close `BacktestTradeRecord` objects.

- [ ] **Step 4: Run focused test and full execution tests**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_backtest_execution.py::test_backtest_trade_records_include_stable_trade_id -q
rtk .venv/bin/python -m pytest tests/test_backtest_execution.py -q
```

- [ ] **Step 5: Commit**

```bash
rtk git add trading_system/backtest_execution.py tests/test_backtest_execution.py
rtk git commit -m "feat: add stable backtest trade identifiers"
```

---

### Task 3: Capture Decision And Regime Snapshots Per Trade

**Files:**
- Modify: `trading_system/trade_audit.py`
- Modify: `trading_system/backtest_execution.py`
- Modify: `backtest_engine.py`
- Test: `tests/test_trade_audit.py`
- Test: `tests/test_backtest_audit_logging.py`

- [ ] **Step 1: Add regime classification tests**

Append to `tests/test_trade_audit.py`:

```python
from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData
from trading_system.trade_audit import build_market_regime_snapshot


def _symbol_market_data(
    *,
    close=100.0,
    sma20=105.0,
    sma50=100.0,
    sma200=90.0,
    rsi14=55.0,
    volatility20=0.025,
    gap_pct=0.012,
    volume_ratio=1.8,
):
    return SymbolMarketData(
        symbol="AAPL",
        market_cap=1000000000,
        close=close,
        high_20d=112.0,
        low_20d=88.0,
        volume=1000000,
        indicators=IndicatorSnapshot(
            atr14=2.0,
            rsi14=rsi14,
            sma20=sma20,
            sma50=sma50,
            sma200=sma200,
            volatility20=volatility20,
            avg_volume20=600000,
        ),
        premarket=PremarketSnapshot(
            latest_price=101.2,
            gap_pct=gap_pct,
            volume=120000,
            timestamp="2026-01-05T09:30:00-05:00",
        ),
        price_summary="test",
        raw_metrics={"volume_ratio": volume_ratio},
    )


def test_build_market_regime_snapshot_tags_symbol_context():
    regime = build_market_regime_snapshot(_symbol_market_data())

    assert regime.market_trend == "uptrend"
    assert regime.volatility_regime == "normal"
    assert regime.gap_regime == "gap_up"
    assert regime.volume_regime == "high_volume"
    assert regime.symbol_momentum_regime == "above_sma20"
    assert "rsi_neutral" in regime.tags
```

- [ ] **Step 2: Add failing backtest audit context test**

Create `tests/test_backtest_audit_logging.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_system.backtest_execution import BacktestExecutionEngine
from trading_system.models import (
    DebateResult,
    IndicatorSnapshot,
    PremarketSnapshot,
    SymbolDebate,
    SymbolMarketData,
    TradeDecision,
)
from trading_system.trade_audit import build_decision_snapshot


def _market(symbol="AAPL"):
    return SymbolMarketData(
        symbol=symbol,
        market_cap=1000000000,
        close=100.0,
        high_20d=110.0,
        low_20d=90.0,
        volume=1000000,
        indicators=IndicatorSnapshot(
            atr14=2.0,
            rsi14=58.0,
            sma20=102.0,
            sma50=98.0,
            sma200=90.0,
            volatility20=0.025,
            avg_volume20=700000,
        ),
        premarket=PremarketSnapshot(
            latest_price=101.0,
            gap_pct=0.01,
            volume=120000,
            timestamp="2026-01-05T09:30:00-05:00",
        ),
        price_summary="trend up",
        score_breakdown={"total": 1.2},
        raw_metrics={"volume_ratio": 1.4},
    )


def _debate(symbol="AAPL"):
    market = _market(symbol)
    return SymbolDebate(
        symbol=symbol,
        market_data=market,
        bull_case=DebateResult(
            symbol=symbol,
            position="bull",
            confidence=0.88,
            arguments=["trend up"],
            risks=["gap fade"],
            key_levels={"support": 98.0},
        ),
        bear_case=DebateResult(
            symbol=symbol,
            position="bear",
            confidence=0.42,
            arguments=["overextended"],
            risks=["short squeeze"],
            key_levels={"resistance": 106.0},
        ),
    )


def test_build_decision_snapshot_preserves_model_audit_fields():
    decision = TradeDecision(
        symbol="AAPL",
        action="long",
        confidence=0.82,
        allocation=0.10,
        raw_confidence=0.94,
        calibrated_confidence=0.82,
        estimated_win_probability=0.66,
        expected_upside_pct=0.09,
        expected_downside_pct=0.035,
        expected_value_pct=0.047,
        risk_reward=2.57,
        evidence_count=5,
        confidence_cap_reason="calibration_cap",
        target_price=112.0,
        invalidation_price=96.0,
        reward_risk_ratio=3.0,
    )

    snapshot = build_decision_snapshot(decision, _debate())

    assert snapshot.raw_confidence == 0.94
    assert snapshot.calibrated_confidence == 0.82
    assert snapshot.bull_arguments == ["trend up"]
    assert snapshot.bear_risks == ["short squeeze"]


def test_backtest_trade_record_includes_decision_and_regime_snapshots():
    engine = BacktestExecutionEngine(initial_cash=100000.0, config=None)
    engine.run_id = "unit-run"
    decision = TradeDecision(
        symbol="AAPL",
        action="long",
        confidence=0.82,
        allocation=0.10,
        raw_confidence=0.94,
        calibrated_confidence=0.82,
        expected_value_pct=0.047,
    )
    now = datetime(2026, 1, 5, tzinfo=UTC)
    market = _market()
    debate = _debate()
    engine.set_decision_contexts(
        current_time=now,
        decisions=[decision],
        debates=[debate],
        selected_symbols=[market],
    )

    engine.process_decisions([decision], {"AAPL": market}, now)
    engine.process_decisions([], {"AAPL": 95.0}, now + timedelta(days=8))

    trade = engine.trades[0]
    assert trade.decision_snapshot["raw_confidence"] == 0.94
    assert trade.regime_snapshot["market_trend"] == "uptrend"
    assert trade.entry_market_snapshot["score_breakdown"]["total"] == 1.2
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_trade_audit.py tests/test_backtest_audit_logging.py -q
```

Expected: fail because snapshot helpers and `set_decision_contexts` do not exist.

- [ ] **Step 4: Implement snapshot helpers**

In `trading_system/trade_audit.py`, add:

```python
from trading_system.models import SymbolDebate, SymbolMarketData, TradeDecision
from trading_system.utils import dataclass_to_dict


def build_decision_snapshot(
    decision: TradeDecision,
    debate: SymbolDebate | None,
) -> DecisionSnapshot:
    return DecisionSnapshot(
        symbol=decision.symbol,
        action=decision.action,
        confidence=decision.confidence,
        raw_confidence=decision.raw_confidence,
        calibrated_confidence=decision.calibrated_confidence,
        allocation=decision.allocation,
        estimated_win_probability=decision.estimated_win_probability,
        expected_upside_pct=decision.expected_upside_pct,
        expected_downside_pct=decision.expected_downside_pct,
        expected_value_pct=decision.expected_value_pct,
        risk_reward=decision.risk_reward,
        evidence_count=decision.evidence_count,
        confidence_cap_reason=decision.confidence_cap_reason,
        target_price=decision.target_price,
        invalidation_price=decision.invalidation_price,
        reward_risk_ratio=decision.reward_risk_ratio,
        reason=decision.catalyst,
        bull_confidence=debate.bull_case.confidence if debate else None,
        bear_confidence=debate.bear_case.confidence if debate else None,
        bull_arguments=list(debate.bull_case.arguments) if debate else [],
        bear_arguments=list(debate.bear_case.arguments) if debate else [],
        bull_risks=list(debate.bull_case.risks) if debate else [],
        bear_risks=list(debate.bear_case.risks) if debate else [],
    )


def build_market_regime_snapshot(market: SymbolMarketData) -> MarketRegimeSnapshot:
    indicators = market.indicators
    tags: list[str] = []
    if indicators.sma20 > indicators.sma50 > indicators.sma200:
        market_trend = "uptrend"
        tags.append("sma20_above_sma50")
    elif indicators.sma20 < indicators.sma50 < indicators.sma200:
        market_trend = "downtrend"
        tags.append("sma20_below_sma50")
    else:
        market_trend = "mixed"
        tags.append("sma_stack_mixed")

    volatility = indicators.volatility20
    if volatility >= 0.05:
        volatility_regime = "high"
    elif volatility <= 0.015:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    gap_pct = market.premarket.gap_pct or 0.0
    if gap_pct >= 0.005:
        gap_regime = "gap_up"
    elif gap_pct <= -0.005:
        gap_regime = "gap_down"
    else:
        gap_regime = "flat_open"

    volume_ratio = market.raw_metrics.get("volume_ratio", 0.0)
    if volume_ratio >= 1.25:
        volume_regime = "high_volume"
    elif volume_ratio and volume_ratio <= 0.75:
        volume_regime = "low_volume"
    else:
        volume_regime = "normal_volume"

    if market.close >= indicators.sma20:
        symbol_momentum_regime = "above_sma20"
    else:
        symbol_momentum_regime = "below_sma20"

    if indicators.rsi14 >= 70:
        tags.append("rsi_overbought")
    elif indicators.rsi14 <= 30:
        tags.append("rsi_oversold")
    else:
        tags.append("rsi_neutral")

    return MarketRegimeSnapshot(
        market_trend=market_trend,
        volatility_regime=volatility_regime,
        gap_regime=gap_regime,
        volume_regime=volume_regime,
        symbol_momentum_regime=symbol_momentum_regime,
        tags=tags,
    )


def compact_market_snapshot(market: SymbolMarketData | None) -> dict[str, Any]:
    if market is None:
        return {}
    payload = dataclass_to_dict(market)
    return {
        "symbol": payload["symbol"],
        "close": payload["close"],
        "high_20d": payload["high_20d"],
        "low_20d": payload["low_20d"],
        "volume": payload["volume"],
        "indicators": payload["indicators"],
        "premarket": payload["premarket"],
        "score_breakdown": payload.get("score_breakdown", {}),
        "raw_metrics": payload.get("raw_metrics", {}),
        "data_quality_flags": payload.get("data_quality_flags", []),
        "is_tradeable": payload.get("is_tradeable", True),
    }
```

- [ ] **Step 5: Add context fields to backtest records**

In `BacktestPosition`, add:

```python
decision_id: str = ""
decision_snapshot: dict[str, Any] = field(default_factory=dict)
regime_snapshot: dict[str, Any] = field(default_factory=dict)
entry_market_snapshot: dict[str, Any] = field(default_factory=dict)
```

In `BacktestTradeRecord`, add:

```python
decision_id: str = ""
decision_snapshot: dict[str, Any] = field(default_factory=dict)
regime_snapshot: dict[str, Any] = field(default_factory=dict)
entry_market_snapshot: dict[str, Any] = field(default_factory=dict)
exit_market_snapshot: dict[str, Any] = field(default_factory=dict)
```

In `BacktestExecutionEngine.__init__`, add:

```python
self._decision_contexts: dict[str, dict[str, Any]] = {}
```

Add method:

```python
def set_decision_contexts(
    self,
    *,
    current_time: datetime,
    decisions: list[TradeDecision],
    debates: list[Any],
    selected_symbols: list[SymbolMarketData],
) -> None:
    from trading_system.trade_audit import (
        build_decision_id,
        build_decision_snapshot,
        build_market_regime_snapshot,
        compact_market_snapshot,
    )

    debate_map = {item.symbol: item for item in debates}
    market_map = {item.symbol: item for item in selected_symbols}
    self._decision_contexts = {}
    for decision in decisions:
        market = market_map.get(decision.symbol)
        debate = debate_map.get(decision.symbol)
        decision_id = build_decision_id(
            run_id=self.run_id,
            symbol=decision.symbol,
            decision_time=current_time.isoformat(),
        )
        self._decision_contexts[decision.symbol] = {
            "decision_id": decision_id,
            "decision_snapshot": build_decision_snapshot(decision, debate).to_dict(),
            "regime_snapshot": build_market_regime_snapshot(market).to_dict() if market else {},
            "entry_market_snapshot": compact_market_snapshot(market),
        }
```

When opening `BacktestPosition`, read:

```python
context = self._decision_contexts.get(dec.symbol, {})
```

Set position fields from that context.

In `_close_position` and `_take_partial_profit`, copy position context onto `BacktestTradeRecord`.

For `exit_market_snapshot`, use `compact_market_snapshot(snapshot)` when closing from a `SymbolMarketData` snapshot and `{ "mark_price": price }` when only a float is available.

- [ ] **Step 6: Pass contexts from the backtest engine**

In `backtest_engine.py`, after `decisions = decision_engine.decide(debates)` and before `execution.process_decisions(...)`, add:

```python
execution.set_decision_contexts(
    current_time=current_day,
    decisions=decisions,
    debates=debates,
    selected_symbols=selected_symbols,
)
```

Set the run id after creating the engine:

```python
execution.run_id = run_path.name
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_trade_audit.py tests/test_backtest_audit_logging.py tests/test_backtest_execution.py -q
```

Commit:

```bash
rtk git add trading_system/trade_audit.py trading_system/backtest_execution.py backtest_engine.py tests/test_trade_audit.py tests/test_backtest_audit_logging.py tests/test_backtest_execution.py
rtk git commit -m "feat: attach decision context to backtest trades"
```

---

### Task 4: Emit Append-Only Backtest Audit Events

**Files:**
- Modify: `trading_system/backtest_execution.py`
- Modify: `backtest_engine.py`
- Test: `tests/test_backtest_audit_logging.py`

- [ ] **Step 1: Add failing audit event test**

Append to `tests/test_backtest_audit_logging.py`:

```python
from trading_system.trade_audit import load_audit_events


def test_backtest_engine_emits_entry_and_exit_audit_events(tmp_path):
    audit_path = tmp_path / "trade_audit_events.jsonl"
    engine = BacktestExecutionEngine(initial_cash=100000.0, config=None)
    engine.run_id = "unit-run"
    engine.audit_log_path = audit_path
    decision = TradeDecision(
        symbol="AAPL",
        action="long",
        confidence=0.82,
        allocation=0.10,
    )
    now = datetime(2026, 1, 5, tzinfo=UTC)

    engine.process_decisions([decision], {"AAPL": 100.0}, now)
    engine.process_decisions([], {"AAPL": 95.0}, now + timedelta(days=8))

    events = load_audit_events(audit_path)
    assert [event.event_type for event in events] == ["entry_fill", "exit_fill"]
    assert events[0].payload["qty"] > 0
    assert events[1].payload["exit_reason"] == "extended_hold"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_backtest_audit_logging.py::test_backtest_engine_emits_entry_and_exit_audit_events -q
```

Expected: fail because audit events are not emitted.

- [ ] **Step 3: Add audit event emission**

In `BacktestExecutionEngine.__init__`, add:

```python
self.audit_log_path: Path | None = None
self._next_audit_event_sequence = 1
```

Add helper:

```python
def _emit_audit_event(
    self,
    *,
    event_type: str,
    occurred_at: datetime,
    symbol: str | None,
    trade_id: str | None,
    decision_id: str | None,
    payload: dict[str, Any],
) -> None:
    if self.audit_log_path is None:
        return
    from trading_system.trade_audit import AuditEvent, append_audit_event

    event = AuditEvent(
        event_id=f"{self.run_id}:{self._next_audit_event_sequence}",
        run_id=self.run_id,
        trade_id=trade_id,
        decision_id=decision_id,
        event_type=event_type,
        occurred_at=occurred_at.isoformat(),
        symbol=symbol,
        payload=payload,
    )
    self._next_audit_event_sequence += 1
    append_audit_event(self.audit_log_path, event)
```

Emit `entry_fill` immediately after opening a position:

```python
self._emit_audit_event(
    event_type="entry_fill",
    occurred_at=current_time,
    symbol=dec.symbol,
    trade_id=trade_id,
    decision_id=context.get("decision_id"),
    payload={
        "side": dec.action,
        "qty": qty,
        "fill_price": round(fill_price, 4),
        "notional": round(qty * fill_price, 2),
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "sizing_reason": sizing_reason,
        "decision_snapshot": context.get("decision_snapshot", {}),
        "regime_snapshot": context.get("regime_snapshot", {}),
    },
)
```

Emit `exit_fill` after appending a full-close trade:

```python
self._emit_audit_event(
    event_type="exit_fill",
    occurred_at=exit_time,
    symbol=record.symbol,
    trade_id=record.trade_id,
    decision_id=record.decision_id,
    payload={
        "side": record.side,
        "qty": record.qty,
        "exit_price": record.exit_price,
        "exit_reason": record.exit_reason,
        "gross_pnl": round(record.gross_pnl, 2),
        "net_pnl": round(record.net_pnl, 2),
        "return_pct": record.return_pct,
        "mfe_pct": record.mfe_pct,
        "mae_pct": record.mae_pct,
    },
)
```

Emit `partial_exit_fill` in `_take_partial_profit`.

- [ ] **Step 4: Wire path into backtest reports**

In `backtest_engine.py`, after creating `run_path`, set:

```python
execution.audit_log_path = run_path / "trade_audit_events.jsonl"
```

In `build_backtest_report`, include:

```python
"trade_audit": {
    "events_path": "trade_audit_events.jsonl",
    "event_count": len(execution.load_audit_events()) if hasattr(execution, "load_audit_events") else None,
},
```

Add method to execution:

```python
def load_audit_events(self) -> list[Any]:
    if self.audit_log_path is None:
        return []
    from trading_system.trade_audit import load_audit_events
    return load_audit_events(self.audit_log_path)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_backtest_audit_logging.py tests/test_backtest_engine.py -q
```

Commit:

```bash
rtk git add trading_system/backtest_execution.py backtest_engine.py tests/test_backtest_audit_logging.py tests/test_backtest_engine.py
rtk git commit -m "feat: write backtest trade audit events"
```

---

### Task 5: Add Live Paper Order Lifecycle Logging

**Files:**
- Modify: `trading_system/execution.py`
- Modify: `trading_system/reporting.py`
- Test: `tests/test_live_audit_logging.py`

- [ ] **Step 1: Write failing live audit tests**

Create `tests/test_live_audit_logging.py`:

```python
from __future__ import annotations

from pathlib import Path

from trading_system.trade_audit import load_audit_events
from trading_system.reporting import write_live_order_audit_events


def test_write_live_order_audit_events_records_submit_and_fill(tmp_path):
    path = tmp_path / "trade_audit_events.jsonl"
    execution_results = [
        {
            "symbol": "AAPL",
            "status": "submitted",
            "order_id": "order-1",
            "client_order_id": "client-1",
            "side": "buy",
            "qty": 10,
            "submitted_at": "2026-04-29T09:31:00-04:00",
            "filled_qty": 10,
            "filled_avg_price": 101.25,
            "broker_status": "filled",
        }
    ]

    write_live_order_audit_events(
        audit_path=path,
        run_id="live-run",
        occurred_at="2026-04-29T09:31:30-04:00",
        execution_results=execution_results,
    )

    events = load_audit_events(path)
    assert len(events) == 1
    assert events[0].event_type == "broker_order_update"
    assert events[0].payload["order_id"] == "order-1"
    assert events[0].payload["filled_avg_price"] == 101.25
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_live_audit_logging.py -q
```

Expected: fail because `write_live_order_audit_events` does not exist.

- [ ] **Step 3: Implement live audit writer**

In `trading_system/reporting.py`, import:

```python
from datetime import UTC, datetime
from trading_system.trade_audit import AuditEvent, append_audit_event
```

Add:

```python
def write_live_order_audit_events(
    *,
    audit_path: Path,
    run_id: str,
    occurred_at: str | None,
    execution_results: list[dict],
) -> None:
    timestamp = occurred_at or datetime.now(UTC).isoformat()
    for index, result in enumerate(execution_results, start=1):
        symbol = result.get("symbol")
        event = AuditEvent(
            event_id=f"{run_id}:broker:{index}",
            run_id=run_id,
            trade_id=result.get("trade_id"),
            decision_id=result.get("decision_id"),
            event_type="broker_order_update",
            occurred_at=timestamp,
            symbol=symbol,
            payload={
                "symbol": symbol,
                "status": result.get("status"),
                "broker_status": result.get("broker_status"),
                "order_id": result.get("order_id"),
                "client_order_id": result.get("client_order_id"),
                "side": result.get("side"),
                "qty": result.get("qty"),
                "filled_qty": result.get("filled_qty"),
                "filled_avg_price": result.get("filled_avg_price"),
                "submitted_at": result.get("submitted_at"),
                "raw": result,
            },
        )
        append_audit_event(audit_path, event)
```

In `write_run_report`, after `write_ai_debug_log(...)`, add:

```python
write_live_order_audit_events(
    audit_path=run_path / "trade_audit_events.jsonl",
    run_id=run_path.name,
    occurred_at=None,
    execution_results=execution_results,
)
```

- [ ] **Step 4: Add broker response fields in execution results**

In `trading_system/execution.py`, find result dictionaries returned by `submit_orders`. Ensure each result includes:

```python
{
    "order_id": getattr(order, "id", None),
    "client_order_id": getattr(order, "client_order_id", None),
    "broker_status": str(getattr(order, "status", "")),
    "submitted_at": str(getattr(order, "submitted_at", "")) if getattr(order, "submitted_at", None) else None,
    "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
    "filled_avg_price": float(getattr(order, "filled_avg_price", 0) or 0),
}
```

For dry-run results, include the same keys with `None` or `0.0`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_live_audit_logging.py tests/test_reporting.py tests/test_execution.py -q
```

If `tests/test_reporting.py` does not exist, run:

```bash
rtk .venv/bin/python -m pytest tests/test_live_audit_logging.py tests/test_execution.py -q
```

Commit:

```bash
rtk git add trading_system/reporting.py trading_system/execution.py tests/test_live_audit_logging.py
rtk git commit -m "feat: log live order lifecycle audit events"
```

---

### Task 6: Add Weekly Review Analytics

**Files:**
- Create: `trading_system/weekly_review.py`
- Create: `scripts/weekly_trade_review.py`
- Test: `tests/test_weekly_review.py`

- [ ] **Step 1: Write failing weekly review tests**

Create `tests/test_weekly_review.py`:

```python
from __future__ import annotations

import json

from trading_system.weekly_review import build_weekly_review


def test_weekly_review_groups_by_regime_and_exit_reason(tmp_path):
    report = {
        "metadata": {
            "start_date": "2026-01-05",
            "end_date": "2026-01-09",
            "status": "completed",
        },
        "performance": {
            "final_equity": 102000,
            "net_pnl": 2000,
            "max_drawdown_pct": -0.012,
            "profit_factor": 1.8,
        },
        "all_trades": [
            {
                "trade_id": "run:2026-01-05:AAPL:long:1",
                "symbol": "AAPL",
                "side": "long",
                "entry_time": "2026-01-05T00:00:00+00:00",
                "exit_reason": "extended_hold",
                "net_pnl": 1200,
                "return_pct": 0.04,
                "confidence": 0.9,
                "mfe_pct": 0.08,
                "mae_pct": -0.01,
                "regime_snapshot": {
                    "market_trend": "uptrend",
                    "volatility_regime": "normal",
                    "gap_regime": "gap_up",
                    "volume_regime": "high_volume",
                    "symbol_momentum_regime": "above_sma20",
                    "tags": ["rsi_neutral"],
                },
            },
            {
                "trade_id": "run:2026-01-06:MSFT:long:2",
                "symbol": "MSFT",
                "side": "long",
                "entry_time": "2026-01-06T00:00:00+00:00",
                "exit_reason": "stop_loss",
                "net_pnl": -300,
                "return_pct": -0.015,
                "confidence": 0.82,
                "mfe_pct": 0.01,
                "mae_pct": -0.03,
                "regime_snapshot": {
                    "market_trend": "mixed",
                    "volatility_regime": "high",
                    "gap_regime": "gap_down",
                    "volume_regime": "normal_volume",
                    "symbol_momentum_regime": "below_sma20",
                    "tags": ["rsi_neutral"],
                },
            },
        ],
    }
    path = tmp_path / "backtest_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    review = build_weekly_review([path])

    assert review["summary"]["trade_count"] == 2
    assert review["summary"]["net_pnl"] == 900
    assert review["by_exit_reason"]["extended_hold"]["net_pnl"] == 1200
    assert review["by_regime"]["market_trend=uptrend"]["trade_count"] == 1
    assert review["adaptation_candidates"][0]["category"] in {
        "exit_reason",
        "regime",
        "confidence",
        "concentration",
    }
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_weekly_review.py -q
```

Expected: fail because `trading_system.weekly_review` does not exist.

- [ ] **Step 3: Implement weekly review module**

Create `trading_system/weekly_review.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_trades(paths: list[Path]) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trades.extend(payload.get("all_trades", []))
    return trades


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if row.get("net_pnl", 0.0) > 0]
    losses = [row for row in rows if row.get("net_pnl", 0.0) < 0]
    gross_win = sum(row.get("net_pnl", 0.0) for row in wins)
    gross_loss = -sum(row.get("net_pnl", 0.0) for row in losses)
    net_pnl = sum(row.get("net_pnl", 0.0) for row in rows)
    return {
        "trade_count": len(rows),
        "net_pnl": round(net_pnl, 2),
        "win_rate": _ratio(len(wins), len(rows)),
        "average_return_pct": round(
            sum(row.get("return_pct", 0.0) for row in rows) / len(rows),
            4,
        )
        if rows
        else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss else None if gross_win else 0.0,
        "average_mfe_pct": round(sum(row.get("mfe_pct", 0.0) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
        "average_mae_pct": round(sum(row.get("mae_pct", 0.0) for row in rows) / len(rows), 4)
        if rows
        else 0.0,
    }


def _group(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return {key: _summary(value) for key, value in sorted(grouped.items())}


def _regime_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        regime = row.get("regime_snapshot") or {}
        for key in (
            "market_trend",
            "volatility_regime",
            "gap_regime",
            "volume_regime",
            "symbol_momentum_regime",
        ):
            value = regime.get(key)
            if value:
                grouped[f"{key}={value}"].append(row)
    return {key: _summary(value) for key, value in sorted(grouped.items())}


def _adaptation_candidates(review: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for reason, stats in review["by_exit_reason"].items():
        if stats["trade_count"] >= 2 and stats["net_pnl"] < 0:
            candidates.append(
                {
                    "category": "exit_reason",
                    "name": reason,
                    "evidence": stats,
                    "suggested_review": "Review stop, thesis-failure, or hold-extension handling for this exit reason.",
                }
            )
    for regime, stats in review["by_regime"].items():
        if stats["trade_count"] >= 2 and stats["profit_factor"] is not None and stats["profit_factor"] < 1.0:
            candidates.append(
                {
                    "category": "regime",
                    "name": regime,
                    "evidence": stats,
                    "suggested_review": "Consider reducing size, requiring higher confidence, or skipping this regime.",
                }
            )
    if not candidates:
        candidates.append(
            {
                "category": "confidence",
                "name": "weekly_confidence_review",
                "evidence": review["summary"],
                "suggested_review": "Compare confidence buckets and raise or lower thresholds only with enough trades.",
            }
        )
    return candidates


def build_weekly_review(report_paths: list[Path]) -> dict[str, Any]:
    trades = _load_trades(report_paths)
    review = {
        "summary": _summary(trades),
        "by_symbol": _group(trades, lambda row: row.get("symbol", "UNKNOWN")),
        "by_side": _group(trades, lambda row: row.get("side", "unknown")),
        "by_exit_reason": _group(trades, lambda row: row.get("exit_reason", "unknown")),
        "by_regime": _regime_groups(trades),
    }
    review["adaptation_candidates"] = _adaptation_candidates(review)
    return review


def render_weekly_review_markdown(review: dict[str, Any]) -> str:
    lines = ["# Weekly Trading Review", ""]
    summary = review["summary"]
    lines.extend(
        [
            f"- Trades: {summary['trade_count']}",
            f"- Net P/L: ${summary['net_pnl']:,.2f}",
            f"- Win rate: {summary['win_rate']:.1%}",
            f"- Profit factor: {summary['profit_factor']}",
            f"- Average return: {summary['average_return_pct']:.2%}",
            "",
            "## Adaptation Candidates",
        ]
    )
    for item in review["adaptation_candidates"]:
        lines.append(
            f"- {item['category']}: {item['name']} - {item['suggested_review']}"
        )
    lines.append("")
    lines.append("## Exit Reasons")
    for reason, stats in review["by_exit_reason"].items():
        lines.append(
            f"- {reason}: trades={stats['trade_count']} net=${stats['net_pnl']:,.2f} "
            f"win={stats['win_rate']:.1%} pf={stats['profit_factor']}"
        )
    lines.append("")
    lines.append("## Regimes")
    for regime, stats in review["by_regime"].items():
        lines.append(
            f"- {regime}: trades={stats['trade_count']} net=${stats['net_pnl']:,.2f} "
            f"win={stats['win_rate']:.1%} pf={stats['profit_factor']}"
        )
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/weekly_trade_review.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_system.weekly_review import build_weekly_review, render_weekly_review_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weekly trading review from backtest reports.")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("weekly_reviews"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review = build_weekly_review(args.reports)
    markdown = render_weekly_review_markdown(review)
    json_path = args.output_dir / "weekly_review.json"
    md_path = args.output_dir / "weekly_review.md"
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_weekly_review.py -q
rtk .venv/bin/python scripts/weekly_trade_review.py backtests/20260429T195150Z/backtest_report.json --output-dir /tmp/weekly_review_check
```

Commit:

```bash
rtk git add trading_system/weekly_review.py scripts/weekly_trade_review.py tests/test_weekly_review.py
rtk git commit -m "feat: add weekly trade review analytics"
```

---

### Task 7: Add Weekly Review Fields For Strategy Adaptation

**Files:**
- Modify: `trading_system/weekly_review.py`
- Modify: `analyze_backtest_correlation.py`
- Test: `tests/test_weekly_review.py`

- [ ] **Step 1: Add tests for adaptation metrics**

Append to `tests/test_weekly_review.py`:

```python
def test_weekly_review_reports_concentration_and_confidence_buckets(tmp_path):
    report = {
        "all_trades": [
            {"symbol": "AAPL", "net_pnl": 900, "return_pct": 0.04, "confidence": 0.91, "exit_reason": "target_hit"},
            {"symbol": "AAPL", "net_pnl": 100, "return_pct": 0.01, "confidence": 0.88, "exit_reason": "extended_hold"},
            {"symbol": "MSFT", "net_pnl": -200, "return_pct": -0.02, "confidence": 0.72, "exit_reason": "stop_loss"},
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    review = build_weekly_review([path])

    assert review["concentration"]["top_symbol"] == "AAPL"
    assert review["concentration"]["top_symbol_profit_share"] == 1.25
    assert review["by_confidence_bucket"]["0.90-1.00"]["net_pnl"] == 900
    assert review["by_confidence_bucket"]["0.70-0.80"]["net_pnl"] == -200
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_weekly_review.py::test_weekly_review_reports_concentration_and_confidence_buckets -q
```

Expected: fail because concentration and confidence buckets are not implemented.

- [ ] **Step 3: Implement concentration and confidence groups**

In `trading_system/weekly_review.py`, add:

```python
CONFIDENCE_BUCKETS = (
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
)


def _confidence_bucket(confidence: float) -> str:
    for lower, upper in CONFIDENCE_BUCKETS:
        if confidence >= lower and (confidence < upper or upper == 1.0):
            return f"{lower:.2f}-{upper:.2f}"
    return "outside_buckets"


def _concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_symbol: dict[str, float] = defaultdict(float)
    for row in rows:
        by_symbol[row.get("symbol", "UNKNOWN")] += row.get("net_pnl", 0.0)
    if not by_symbol:
        return {"top_symbol": None, "top_symbol_net_pnl": 0.0, "top_symbol_profit_share": 0.0}
    top_symbol, top_pnl = max(by_symbol.items(), key=lambda item: item[1])
    total_net = sum(by_symbol.values())
    share = round(top_pnl / total_net, 4) if total_net else 0.0
    return {
        "top_symbol": top_symbol,
        "top_symbol_net_pnl": round(top_pnl, 2),
        "top_symbol_profit_share": share,
    }
```

Add to `build_weekly_review`:

```python
"by_confidence_bucket": _group(
    trades,
    lambda row: _confidence_bucket(float(row.get("confidence", 0.0) or 0.0)),
),
"concentration": _concentration(trades),
```

Update `_adaptation_candidates` to add:

```python
concentration = review.get("concentration", {})
if concentration.get("top_symbol_profit_share", 0.0) > 0.50:
    candidates.append(
        {
            "category": "concentration",
            "name": concentration["top_symbol"],
            "evidence": concentration,
            "suggested_review": "Review whether profits are too dependent on one symbol before increasing size.",
        }
    )
```

- [ ] **Step 4: Update correlation script**

In `analyze_backtest_correlation.py`, after printing confidence buckets, add:

```python
weekly_fields = data.get("weekly_review")
if weekly_fields:
    print("Weekly review fields present.")
```

Also update the fallback branch to prefer `decision_snapshot.raw_confidence` and `decision_snapshot.calibrated_confidence` when present:

```python
raw_confidences = [
    (t.get("decision_snapshot") or {}).get("raw_confidence", t.get("confidence", 0.0))
    for t in trades
]
calibrated_confidences = [
    (t.get("decision_snapshot") or {}).get("calibrated_confidence", t.get("confidence", 0.0))
    for t in trades
]
print(f"Average raw confidence: {np.mean(raw_confidences):.4f}")
print(f"Average calibrated confidence: {np.mean(calibrated_confidences):.4f}")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_weekly_review.py -q
```

Commit:

```bash
rtk git add trading_system/weekly_review.py analyze_backtest_correlation.py tests/test_weekly_review.py
rtk git commit -m "feat: add weekly adaptation metrics"
```

---

### Task 8: Validate With Existing January Reports

**Files:**
- No source changes expected

- [ ] **Step 1: Run full unit tests**

Run:

```bash
rtk .venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Generate a weekly review from the best completed report**

Run:

```bash
rtk .venv/bin/python scripts/weekly_trade_review.py backtests/20260429T195150Z/backtest_report.json --output-dir weekly_reviews/2026-01-hold-partial
```

Expected output:

```text
Wrote weekly_reviews/2026-01-hold-partial/weekly_review.md
Wrote weekly_reviews/2026-01-hold-partial/weekly_review.json
```

- [ ] **Step 3: Inspect the weekly review**

Run:

```bash
sed -n '1,220p' weekly_reviews/2026-01-hold-partial/weekly_review.md
```

Confirm the review includes:

- total trades
- net P/L
- profit factor
- exit-reason breakdown
- regime breakdown when enriched reports are used
- adaptation candidates
- concentration warning if top-symbol dependence is high

- [ ] **Step 4: Commit generated review only if the team wants examples checked in**

If example reviews should be committed:

```bash
rtk git add weekly_reviews/2026-01-hold-partial/weekly_review.md weekly_reviews/2026-01-hold-partial/weekly_review.json
rtk git commit -m "docs: add january hold partial weekly review"
```

If generated reviews should remain local artifacts:

```bash
rtk git status --short
```

Expected: generated `weekly_reviews/` files are untracked or ignored.

---

## Weekly Review Questions This Enables

The final system should make these questions answerable every week without manual log digging:

- Which regimes made money and which regimes lost money?
- Did high confidence actually outperform lower confidence?
- Were losses driven by entries, sizing, stops, thesis-failure exits, or partial exits?
- Did partial exits improve or cap returns?
- Did conditional hold extensions recover losing trades or delay exits?
- Which symbols or sectors dominated returns?
- Are profits concentrated enough that the strategy is brittle?
- Did live fills differ materially from backtest fills?
- Were broker rejects, partial fills, or cancel/replaces a meaningful drag?
- Were JSON retries concentrated in losing trades or low-quality decisions?
- Which one or two strategy parameters deserve a controlled A/B backtest next week?

## Acceptance Criteria

- Every new backtest trade has a stable `trade_id`.
- Every enriched trade includes decision, market, and regime snapshots.
- Backtests write append-only `trade_audit_events.jsonl`.
- Live paper runs write broker lifecycle audit events.
- Weekly review CLI produces both markdown and JSON.
- Weekly review includes summary, exit-reason, confidence, symbol concentration, and regime breakdowns.
- Existing reports remain readable by old analysis scripts.
- Full test suite passes.

## Self-Review Notes

- Spec coverage: the plan covers backtest trade logs, live paper order logs, regime tagging, weekly adaptation summaries, and compatibility with existing reports.
- Placeholder scan: the plan avoids deferred implementation placeholders and gives concrete file paths, commands, and code snippets.
- Type consistency: `trade_id`, `decision_id`, `DecisionSnapshot`, `MarketRegimeSnapshot`, and `AuditEvent` are defined before downstream tasks use them.
