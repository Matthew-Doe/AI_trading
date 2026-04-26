# Confidence Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate decision confidence so it estimates the probability that a `long`, `short`, or `skip` decision is correct over the next 3 trading days.

**Architecture:** Add a read-only confidence calibration module that labels historical run decisions from prior artifacts plus forward close data, computes conservative bucket hit rates, and rescales raw decision confidence before execution sizing uses it. Reuse `MarketDataService` for forward-close lookup, keep the LLM prompts unchanged, and expose a small report surface for inspection.

**Tech Stack:** Python, dataclasses, pytest, existing market data service, existing run artifacts under `runs/`

---

## File Structure

- Create: `trading_system/confidence_calibration.py`
- Modify: `trading_system/config.py`
- Modify: `trading_system/data.py`
- Modify: `trading_system/decision.py`
- Modify: `trading_system/main.py`
- Modify: `.env.example`
- Modify: `README.md`
- Create: `tests/test_confidence_calibration.py`
- Modify: `tests/test_decision.py`
- Modify: `tests/test_config.py`

### Task 1: Add Config And Market-Data Primitives

**Files:**
- Modify: `trading_system/config.py`
- Modify: `trading_system/data.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config and market-data tests**

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_trading_config_defaults_cover_confidence_calibration(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_ACTIONABLE_MOVE_PCT", raising=False)

    config = TradingConfig()

    assert config.confidence_actionable_move_pct == 0.02


def test_fetch_forward_close_window_uses_next_three_trading_days():
    service = MarketDataService.__new__(MarketDataService)
    service.config = TradingConfig()
    service.logger = DummyLogger()
    service._fetch_daily_bars = lambda symbol: pd.DataFrame(
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

    assert reference_close == 101.0
    assert forward_close == 103.0
    assert forward_as_of.startswith("2026-04-24")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL because `TradingConfig` has no `confidence_actionable_move_pct` and `MarketDataService` has no `fetch_forward_close_window`.

- [ ] **Step 3: Add the minimal config and market-data implementation**

```python
# trading_system/config.py
confidence_actionable_move_pct: float = float(
    os.getenv("CONFIDENCE_ACTIONABLE_MOVE_PCT", "0.02")
)
```

```python
# trading_system/data.py
def fetch_forward_close_window(
    self,
    symbol: str,
    *,
    as_of: datetime,
    trading_days_ahead: int = 3,
) -> tuple[float, float, str]:
    daily = self._fetch_daily_bars(symbol)
    closes = daily["Close"].dropna().copy()
    if closes.empty:
        raise DataIngestionError(f"No close history returned for {symbol}")

    market_tz = ZoneInfo(self.config.market_timezone)
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC").tz_convert(market_tz)
    else:
        closes.index = closes.index.tz_convert(market_tz)

    closes = closes.sort_index()
    as_of_local = as_of.astimezone(market_tz)
    eligible = closes[closes.index.date >= as_of_local.date()]
    if len(eligible) <= trading_days_ahead:
        raise DataIngestionError(
            f"Insufficient forward close history for {symbol} at {as_of_local.date()}"
        )

    reference_close = float(eligible.iloc[0])
    forward_close = float(eligible.iloc[trading_days_ahead])
    forward_as_of = eligible.index[trading_days_ahead].isoformat()
    return reference_close, forward_close, forward_as_of
```

```dotenv
# .env.example
CONFIDENCE_ACTIONABLE_MOVE_PCT=0.02
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS

### Task 2: Build Historical Labeling And Bucket Calibration

**Files:**
- Create: `trading_system/confidence_calibration.py`
- Create: `tests/test_confidence_calibration.py`
- Test: `tests/test_confidence_calibration.py`

- [ ] **Step 1: Write the failing labeling and bucket tests**

```python
from trading_system.confidence_calibration import (
    ConfidenceBucket,
    label_decision_correctness,
    calibrate_confidence,
)


def test_label_decision_correctness_for_long_short_and_skip():
    assert label_decision_correctness("long", 0.03, actionable_move_pct=0.02) is True
    assert label_decision_correctness("short", -0.04, actionable_move_pct=0.02) is True
    assert label_decision_correctness("skip", 0.01, actionable_move_pct=0.02) is True
    assert label_decision_correctness("skip", -0.03, actionable_move_pct=0.02) is False


def test_calibrate_confidence_uses_bucket_hit_rate():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=5,
            correct_count=4,
            hit_rate=0.8,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.84,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert calibrated == 0.8


def test_calibrate_confidence_falls_back_when_bucket_is_sparse():
    buckets = [
        ConfidenceBucket(
            lower=0.8,
            upper=0.9,
            sample_count=1,
            correct_count=1,
            hit_rate=1.0,
        )
    ]

    calibrated = calibrate_confidence(
        raw_confidence=0.84,
        buckets=buckets,
        global_hit_rate=0.55,
        minimum_samples=3,
    )

    assert round(calibrated, 2) == 0.70
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/python -m pytest tests/test_confidence_calibration.py -q`
Expected: FAIL because the calibration module does not exist.

- [ ] **Step 3: Implement the calibration module**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConfidenceBucket:
    lower: float
    upper: float
    sample_count: int
    correct_count: int
    hit_rate: float

    def matches(self, confidence: float) -> bool:
        if self.upper >= 1.0:
            return self.lower <= confidence <= self.upper
        return self.lower <= confidence < self.upper


def label_decision_correctness(
    action: str,
    forward_return: float,
    *,
    actionable_move_pct: float,
) -> bool:
    normalized_action = action.lower()
    if normalized_action == "long":
        return forward_return >= actionable_move_pct
    if normalized_action == "short":
        return forward_return <= -actionable_move_pct
    if normalized_action == "skip":
        return abs(forward_return) < actionable_move_pct
    raise ValueError(f"Unsupported action: {action}")


def calibrate_confidence(
    *,
    raw_confidence: float,
    buckets: list[ConfidenceBucket],
    global_hit_rate: float,
    minimum_samples: int,
) -> float:
    matched = next((bucket for bucket in buckets if bucket.matches(raw_confidence)), None)
    if matched is None:
        return raw_confidence
    if matched.sample_count >= minimum_samples:
        return matched.hit_rate
    return round((raw_confidence + global_hit_rate) / 2, 4)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/python -m pytest tests/test_confidence_calibration.py -q`
Expected: PASS

### Task 3: Load Historical Decisions And Integrate Calibration Into DecisionEngine

**Files:**
- Modify: `trading_system/confidence_calibration.py`
- Modify: `trading_system/decision.py`
- Modify: `tests/test_decision.py`
- Test: `tests/test_decision.py tests/test_confidence_calibration.py`

- [ ] **Step 1: Write the failing history-loading and decision integration tests**

```python
from datetime import UTC, datetime

from trading_system.confidence_calibration import build_historical_decision_outcomes
from trading_system.config import TradingConfig
from trading_system.decision import DecisionEngine


def test_build_historical_decision_outcomes_skips_incomplete_forward_window(tmp_path):
    run_dir = tmp_path / "runs" / "20260422T163001Z"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "runs",
        market_data_service=None,
        actionable_move_pct=0.02,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert outcomes == []


def test_validate_and_normalize_applies_confidence_calibration():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger())
    engine._calibrate_confidence = lambda symbol, action, confidence: 0.61  # type: ignore[method-assign]

    payload = {
        "decisions": [
            {"symbol": "AAA", "action": "long", "confidence": 0.88, "allocation": 0.8},
        ]
    }

    decisions = engine._validate_and_normalize(payload, generation_probability=0.95)

    assert decisions[0].confidence == 0.61
    assert decisions[0].action == "long"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `rtk .venv/bin/python -m pytest tests/test_decision.py tests/test_confidence_calibration.py -q`
Expected: FAIL because the historical-outcome loader and calibration hook are not implemented.

- [ ] **Step 3: Implement historical run parsing and DecisionEngine calibration**

```python
# trading_system/confidence_calibration.py
@dataclass(slots=True)
class HistoricalDecisionOutcome:
    run_id: str
    symbol: str
    action: str
    raw_confidence: float
    forward_return: float
    is_correct: bool
    forward_as_of: str


def build_historical_decision_outcomes(
    *,
    run_root: Path,
    market_data_service,
    actionable_move_pct: float,
    now: datetime,
) -> list[HistoricalDecisionOutcome]:
    outcomes: list[HistoricalDecisionOutcome] = []
    for run_path in sorted(run_root.glob("*/decisions.json")):
        run_id = run_path.parent.name
        run_started_at = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if now <= run_started_at + timedelta(days=3):
            continue
        decisions = read_json(run_path)
        for item in decisions:
            if item["action"] == "skip" and item["confidence"] == 0.0:
                raw_confidence = 0.0
            else:
                raw_confidence = float(item["confidence"])
            reference_close, forward_close, forward_as_of = (
                market_data_service.fetch_forward_close_window(
                    item["symbol"],
                    as_of=run_started_at,
                    trading_days_ahead=3,
                )
            )
            forward_return = (
                (forward_close - reference_close) / reference_close
                if reference_close
                else 0.0
            )
            outcomes.append(
                HistoricalDecisionOutcome(
                    run_id=run_id,
                    symbol=item["symbol"],
                    action=item["action"],
                    raw_confidence=raw_confidence,
                    forward_return=forward_return,
                    is_correct=label_decision_correctness(
                        item["action"],
                        forward_return,
                        actionable_move_pct=actionable_move_pct,
                    ),
                    forward_as_of=forward_as_of,
                )
            )
    return outcomes
```

```python
# trading_system/decision.py
from trading_system.confidence_calibration import ConfidenceCalibrator
from trading_system.data import MarketDataService


class DecisionEngine:
    def __init__(self, config: TradingConfig, logger, token_tracker: TokenUsageTracker | None = None):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.token_tracker = token_tracker
        self._last_generation_probability: float | None = None
        self.market_data_service = MarketDataService(config, logger)
        self.confidence_calibrator = ConfidenceCalibrator(
            config=config,
            logger=logger,
            market_data_service=self.market_data_service,
        )

    def _calibrate_confidence(self, symbol: str, action: str, raw_confidence: float) -> float:
        if action == "skip" and raw_confidence <= 0:
            return 0.0
        return self.confidence_calibrator.calibrate(
            symbol=symbol,
            action=action,
            raw_confidence=raw_confidence,
        )
```

```python
# trading_system/decision.py inside _validate_and_normalize
raw_confidence = clamp(min(float(item["confidence"]), confidence_cap), 0.0, 1.0)
calibrated_confidence = self._calibrate_confidence(
    str(item["symbol"]).upper(),
    action,
    raw_confidence,
)
decisions.append(
    TradeDecision(
        symbol=str(item["symbol"]).upper(),
        action=action,
        confidence=calibrated_confidence,
        allocation=clamp(float(item["allocation"]), 0.0, 1.0),
    )
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/python -m pytest tests/test_decision.py tests/test_confidence_calibration.py -q`
Expected: PASS

### Task 4: Emit Calibration Artifacts And Add A Report Surface

**Files:**
- Modify: `trading_system/confidence_calibration.py`
- Modify: `trading_system/main.py`
- Modify: `README.md`
- Test: `tests/test_confidence_calibration.py tests/test_decision.py`

- [ ] **Step 1: Write the failing reporting test**

```python
from trading_system.confidence_calibration import render_calibration_report


def test_render_calibration_report_lists_buckets_and_skips():
    report = render_calibration_report(
        bucket_rows=[
            {"range": "0.80-0.89", "samples": 5, "correct": 4, "hit_rate": 0.8},
        ],
        skipped_samples=[
            {"run_id": "20260422T163001Z", "symbol": "AAPL", "reason": "insufficient forward window"},
        ],
    )

    assert "0.80-0.89" in report
    assert "samples=5" in report
    assert "insufficient forward window" in report
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `rtk .venv/bin/python -m pytest tests/test_confidence_calibration.py -q`
Expected: FAIL because the reporting helper does not exist yet.

- [ ] **Step 3: Implement artifact/report output and CLI wiring**

```python
# trading_system/confidence_calibration.py
def render_calibration_report(*, bucket_rows: list[dict], skipped_samples: list[dict]) -> str:
    lines = ["Confidence calibration report", "Buckets:"]
    for row in bucket_rows:
        lines.append(
            f"- {row['range']}: samples={row['samples']} correct={row['correct']} hit_rate={row['hit_rate']:.2f}"
        )
    if skipped_samples:
        lines.append("Skipped samples:")
        for sample in skipped_samples[:10]:
            lines.append(
                f"- {sample['run_id']} {sample['symbol']}: {sample['reason']}"
            )
    return "\n".join(lines)
```

```python
# trading_system/main.py
parser.add_argument(
    "--confidence-calibration-report",
    action="store_true",
    help="Print the confidence calibration report without running the trading pipeline.",
)
```

```python
# trading_system/main.py
if args.confidence_calibration_report:
    logger = get_logger(config.log_dir, "confidence_calibration")
    calibrator = ConfidenceCalibrator(
        config=config,
        logger=logger,
        market_data_service=MarketDataService(config, logger),
    )
    print(calibrator.build_report())
    return 0
```

```md
# README.md
python -m trading_system.main --confidence-calibration-report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `rtk .venv/bin/python -m pytest tests/test_confidence_calibration.py tests/test_decision.py tests/test_config.py -q`
Expected: PASS

## Self-Review

- Spec coverage: Task 1 covers explicit configuration and forward-close data lookup. Task 2 covers labeling and bucket calibration. Task 3 covers historical run loading, sparse-sample-safe calibration use, and decision-engine integration. Task 4 covers artifacts and a lightweight report surface.
- Placeholder scan: no `TBD`, `TODO`, or unresolved references remain in the plan.
- Type consistency: the plan consistently uses `ConfidenceBucket`, `HistoricalDecisionOutcome`, `ConfidenceCalibrator`, `fetch_forward_close_window`, and `render_calibration_report`.
