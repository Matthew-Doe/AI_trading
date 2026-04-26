from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_system.data import DataIngestionError
from trading_system.utils import read_json


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


@dataclass(slots=True)
class HistoricalDecisionOutcome:
    run_id: str
    symbol: str
    action: str
    raw_confidence: float
    forward_return: float
    is_correct: bool
    forward_as_of: str


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
    
    # If we have enough data for a bucket, blend it 50/50 with the LLM's raw score
    # This prevents the LLM from being 'muted' while still applying a historical reality check.
    if matched and matched.sample_count >= minimum_samples:
        return round((raw_confidence * 0.5) + (matched.hit_rate * 0.5), 2)
    
    # If no bucket, blend with the global hit rate
    if global_hit_rate > 0:
        return round((raw_confidence * 0.7) + (global_hit_rate * 0.3), 2)
        
    return raw_confidence


def build_historical_decision_outcomes(
    *,
    run_root: Path,
    market_data_service: Any,
    actionable_move_pct: float,
    now: datetime,
) -> list[HistoricalDecisionOutcome]:
    outcomes: list[HistoricalDecisionOutcome] = []
    for decisions_path in sorted(run_root.glob("*/decisions.json")):
        run_id = decisions_path.parent.name
        run_started_at = _parse_run_started_at(run_id)
        if run_started_at is None:
            continue
        if run_started_at >= now:
            continue

        decisions = read_json(decisions_path)
        if not isinstance(decisions, list):
            continue

        for item in decisions:
            try:
                reference_close, forward_close, forward_as_of = (
                    market_data_service.fetch_forward_close_window(
                        str(item["symbol"]).upper(),
                        as_of=run_started_at,
                        trading_days_ahead=3,
                    )
                )
            except DataIngestionError as exc:
                if "Insufficient forward close history" in str(exc):
                    continue
                raise

            action = str(item["action"]).lower()
            raw_confidence = float(item.get("confidence", 0.0))
            forward_return = (
                (forward_close - reference_close) / reference_close if reference_close else 0.0
            )
            outcomes.append(
                HistoricalDecisionOutcome(
                    run_id=run_id,
                    symbol=str(item["symbol"]).upper(),
                    action=action,
                    raw_confidence=raw_confidence,
                    forward_return=forward_return,
                    is_correct=label_decision_correctness(
                        action,
                        forward_return,
                        actionable_move_pct=actionable_move_pct,
                    ),
                    forward_as_of=forward_as_of,
                )
            )

    return outcomes


class ConfidenceCalibrator:
    def __init__(
        self,
        *,
        config,
        logger,
        market_data_service,
        run_root: Path | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.market_data_service = market_data_service
        self.run_root = run_root or config.run_dir
        self.minimum_samples = 3
        self.now = now
        self._loaded = False
        self._buckets: list[ConfidenceBucket] = []
        self._global_hit_rate = 0.0

    def calibrate(self, *, symbol: str, action: str, raw_confidence: float) -> float:
        del symbol
        del action
        self._ensure_loaded()
        if not self._buckets:
            return raw_confidence
        return calibrate_confidence(
            raw_confidence=raw_confidence,
            buckets=self._buckets,
            global_hit_rate=self._global_hit_rate,
            minimum_samples=self.minimum_samples,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            outcomes = build_historical_decision_outcomes(
                run_root=self.run_root,
                market_data_service=self.market_data_service,
                actionable_move_pct=self.config.confidence_actionable_move_pct,
                now=self.now or datetime.now(UTC),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Confidence calibration unavailable: %s", exc)
            return

        if not outcomes:
            return

        self._global_hit_rate = sum(outcome.is_correct for outcome in outcomes) / len(outcomes)
        self._buckets = _build_confidence_buckets(outcomes)


def _parse_run_started_at(run_id: str) -> datetime | None:
    for pattern in ("%Y%m%dT%H%M%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(run_id, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _build_confidence_buckets(
    outcomes: list[HistoricalDecisionOutcome],
) -> list[ConfidenceBucket]:
    buckets: list[ConfidenceBucket] = []
    for index in range(10):
        lower = index / 10
        upper = 1.0 if index == 9 else (index + 1) / 10
        bucket = ConfidenceBucket(
            lower=lower,
            upper=upper,
            sample_count=0,
            correct_count=0,
            hit_rate=0.0,
        )
        samples = [outcome for outcome in outcomes if bucket.matches(outcome.raw_confidence)]
        if not samples:
            continue
        correct_count = sum(outcome.is_correct for outcome in samples)
        buckets.append(
            ConfidenceBucket(
                lower=lower,
                upper=upper,
                sample_count=len(samples),
                correct_count=correct_count,
                hit_rate=correct_count / len(samples),
            )
        )
    return buckets
