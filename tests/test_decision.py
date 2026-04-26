from datetime import UTC, datetime
from types import SimpleNamespace

from trading_system.data import DataIngestionError
from trading_system.confidence_calibration import build_historical_decision_outcomes
from trading_system.config import TradingConfig
from trading_system.decision import DecisionEngine, DecisionError


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class IdentityCalibrator:
    def calibrate(self, *, symbol: str, action: str, raw_confidence: float) -> float:
        del symbol
        del action
        return raw_confidence


class StubCalibrator:
    def __init__(self, calibrated_confidence: float):
        self.calibrated_confidence = calibrated_confidence
        self.calls: list[tuple[str, str, float]] = []

    def calibrate(self, *, symbol: str, action: str, raw_confidence: float) -> float:
        self.calls.append((symbol, action, raw_confidence))
        return self.calibrated_confidence


def _dummy_debate(symbol: str = "AAA"):
    indicators = SimpleNamespace(rsi14=55.0, sma20=101.0, sma50=100.0, sma200=95.0)
    premarket = SimpleNamespace(latest_price=102.0, gap_pct=1.2, volume=100000)
    market_data = SimpleNamespace(
        price_summary=f"{symbol} compact summary",
        indicators=indicators,
        premarket=premarket,
    )
    bull_case = SimpleNamespace(confidence=0.82, arguments=["breakout"])
    bear_case = SimpleNamespace(confidence=0.35, arguments=["extended"])
    return SimpleNamespace(
        symbol=symbol,
        market_data=market_data,
        bull_case=bull_case,
        bear_case=bear_case,
    )


def test_decision_normalization_and_thresholding():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    payload = {
        "decisions": [
            {"symbol": "AAA", "action": "long", "confidence": 0.9, "allocation": 0.8},
            {"symbol": "BBB", "action": "short", "confidence": 0.8, "allocation": 0.6},
            {"symbol": "CCC", "action": "long", "confidence": 0.3, "allocation": 0.2},
        ]
    }
    decisions = engine._validate_and_normalize(payload)
    active = [item for item in decisions if item.action != "skip"]
    assert round(sum(item.allocation for item in active), 6) == 1.0
    assert next(item for item in decisions if item.symbol == "CCC").action == "skip"


def test_decide_uses_single_symbol_generation_without_batch_probe():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    debate = _dummy_debate("AAA")

    def fail_if_batch_called(prompt):
        del prompt
        raise AssertionError("batch decision generation should not be used")

    engine._generate_valid_decision_json = fail_if_batch_called  # type: ignore[method-assign]
    engine._build_single_symbol_prompt = lambda debate: "prompt"  # type: ignore[method-assign]
    engine._generate_valid_single_decision_json = (  # type: ignore[method-assign]
        lambda prompt, symbol: {
            "symbol": symbol,
            "action": "long",
            "confidence": 0.8,
            "target_price": 110.0,
            "invalidation_price": 96.0,
            "reward_risk_ratio": 2.5,
        }
    )

    decisions = engine.decide([debate])

    assert len(decisions) == 1
    assert decisions[0].symbol == "AAA"
    assert decisions[0].action == "long"


def test_validate_and_normalize_wraps_single_decision_object():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    payload = {
        "symbol": "AAA",
        "action": "long",
        "confidence": 0.9,
        "allocation": 0.8,
    }

    decisions = engine._validate_and_normalize(payload)

    assert len(decisions) == 1
    assert decisions[0].symbol == "AAA"
    assert decisions[0].action == "long"


def test_decide_per_symbol_skips_symbol_after_repeated_json_failures():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())

    class Debate:
        symbol = "AAPL"

    engine._build_single_symbol_prompt = lambda debate: "prompt"  # type: ignore[method-assign]
    engine._generate_valid_single_decision_json = (  # type: ignore[method-assign]
        lambda prompt, symbol: (_ for _ in ()).throw(DecisionError("bad json"))
    )

    decisions = engine._decide_per_symbol([Debate()])

    assert len(decisions) == 1
    assert decisions[0].symbol == "AAPL"
    assert decisions[0].action == "skip"
    assert decisions[0].confidence == 0.0


def test_validate_and_normalize_caps_confidence_with_generation_probability():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    payload = {
        "decisions": [
            {"symbol": "AAA", "action": "long", "confidence": 0.9, "allocation": 0.8},
        ]
    }

    decisions = engine._validate_and_normalize(payload, generation_probability=0.55)

    assert decisions[0].action == "skip"
    assert decisions[0].confidence == 0.55


def test_generate_valid_single_decision_json_accepts_decisions_wrapper():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    engine._ollama_generate = (  # type: ignore[method-assign]
        lambda prompt, temperature: (
            '{"decisions":[{"symbol":"AAA","action":"long","confidence":0.8,'
            '"target_price":110.0,"invalidation_price":96.0,"reward_risk_ratio":2.5}]}'
        )
    )

    payload = engine._generate_valid_single_decision_json("prompt", "AAA")

    assert payload["symbol"] == "AAA"
    assert payload["action"] == "long"


def test_extract_json_handles_preamble_and_trailing_text():
    raw = "thinking... {\"decisions\": [{\"symbol\": \"AAA\", \"action\": \"skip\", \"confidence\": 0.0}]} trailing"

    payload = DecisionEngine._extract_json(raw)

    assert payload["decisions"][0]["symbol"] == "AAA"


def test_build_historical_decision_outcomes_skips_incomplete_forward_window(tmp_path):
    run_dir = tmp_path / "runs" / "20260422T163001Z"
    run_dir.mkdir(parents=True)
    (run_dir / "decisions.json").write_text(
        '[{"symbol":"AAPL","action":"long","confidence":0.82,"allocation":0.4}]',
        encoding="utf-8",
    )

    class StubMarketDataService:
        def __init__(self):
            self.calls = 0

        def fetch_forward_close_window(self, symbol: str, *, as_of: datetime, trading_days_ahead: int):
            self.calls += 1
            assert symbol == "AAPL"
            assert as_of == datetime(2026, 4, 22, 16, 30, 1, tzinfo=UTC)
            assert trading_days_ahead == 3
            raise DataIngestionError("Insufficient forward close history for AAPL at 2026-04-22")

    market_data_service = StubMarketDataService()

    outcomes = build_historical_decision_outcomes(
        run_root=tmp_path / "runs",
        market_data_service=market_data_service,
        actionable_move_pct=0.02,
        now=datetime(2026, 4, 23, tzinfo=UTC),
    )

    assert outcomes == []
    assert market_data_service.calls == 1


def test_validate_and_normalize_applies_confidence_calibration():
    config = TradingConfig(min_confidence=0.6)
    calibrator = StubCalibrator(0.61)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=calibrator)
    payload = {
        "decisions": [
            {"symbol": "AAA", "action": "long", "confidence": 0.88, "allocation": 0.8},
        ]
    }

    decisions = engine._validate_and_normalize(payload, generation_probability=0.95)

    assert decisions[0].confidence == 0.61
    assert decisions[0].action == "long"
    assert calibrator.calls == [("AAA", "long", 0.88)]


def test_validate_and_normalize_requires_minimum_reward_risk_for_actionable_trade():
    config = TradingConfig(min_confidence=0.6, min_reward_risk_ratio=1.5)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    payload = {
        "decisions": [
            {
                "symbol": "AAA",
                "action": "long",
                "confidence": 0.9,
                "allocation": 0.5,
                "target_price": 105.0,
                "invalidation_price": 98.0,
                "reward_risk_ratio": 1.0,
            },
        ]
    }

    decisions = engine._validate_and_normalize(payload)

    assert decisions[0].action == "skip"
    assert decisions[0].allocation == 0.0


def test_validate_and_normalize_keeps_structured_trade_fields():
    config = TradingConfig(min_confidence=0.6, min_reward_risk_ratio=1.5)
    engine = DecisionEngine(config, DummyLogger(), confidence_calibrator=IdentityCalibrator())
    payload = {
        "decisions": [
            {
                "symbol": "AAA",
                "action": "long",
                "confidence": 0.9,
                "allocation": 0.5,
                "expected_move_pct": 0.06,
                "target_price": 110.0,
                "invalidation_price": 96.0,
                "time_horizon": "3 trading days",
                "catalyst": "volume breakout",
                "reward_risk_ratio": 2.5,
            },
        ]
    }

    decisions = engine._validate_and_normalize(payload)

    assert decisions[0].action == "long"
    assert decisions[0].target_price == 110.0
    assert decisions[0].invalidation_price == 96.0
    assert decisions[0].reward_risk_ratio == 2.5
