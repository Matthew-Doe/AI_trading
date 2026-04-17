from trading_system.config import TradingConfig
from trading_system.decision import DecisionEngine, DecisionError


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_decision_normalization_and_thresholding():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger())
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


def test_decide_per_symbol_skips_symbol_after_repeated_json_failures():
    config = TradingConfig(min_confidence=0.6)
    engine = DecisionEngine(config, DummyLogger())

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
    engine = DecisionEngine(config, DummyLogger())
    payload = {
        "decisions": [
            {"symbol": "AAA", "action": "long", "confidence": 0.9, "allocation": 0.8},
        ]
    }

    decisions = engine._validate_and_normalize(payload, generation_probability=0.55)

    assert decisions[0].action == "skip"
    assert decisions[0].confidence == 0.55
