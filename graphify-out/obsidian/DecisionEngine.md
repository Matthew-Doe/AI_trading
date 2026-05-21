---
source_file: "trading_system/decision.py"
type: "code"
community: "Decision Engine Tests"
location: "L20"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Decision_Engine_Tests
---

# DecisionEngine

## Connections
- [[.__init__()_9]] - `method` [EXTRACTED]
- [[._build_decision_prompt()]] - `method` [EXTRACTED]
- [[._build_single_symbol_prompt()]] - `method` [EXTRACTED]
- [[._calibrate_confidence()]] - `method` [EXTRACTED]
- [[._confidence_cap_reason()]] - `method` [EXTRACTED]
- [[._decide_per_symbol()]] - `method` [EXTRACTED]
- [[._generate_valid_decision_json()]] - `method` [EXTRACTED]
- [[._generate_valid_single_decision_json()]] - `method` [EXTRACTED]
- [[._validate_and_normalize()]] - `method` [EXTRACTED]
- [[.decide()]] - `method` [EXTRACTED]
- [[ConfidenceCalibrator]] - `uses` [INFERRED]
- [[DummyLogger_6]] - `uses` [INFERRED]
- [[IdentityCalibrator]] - `uses` [INFERRED]
- [[LLMClient]] - `uses` [INFERRED]
- [[MarketDataService]] - `uses` [INFERRED]
- [[StubCalibrator]] - `uses` [INFERRED]
- [[SymbolDebate]] - `uses` [INFERRED]
- [[TokenUsageTracker]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[decision.py]] - `contains` [EXTRACTED]
- [[manual_check()]] - `calls` [INFERRED]
- [[run_backtest()]] - `calls` [INFERRED]
- [[run_pipeline()]] - `calls` [INFERRED]
- [[test_decide_per_symbol_skips_symbol_after_repeated_json_failures()]] - `calls` [INFERRED]
- [[test_decide_uses_single_symbol_generation_without_batch_probe()]] - `calls` [INFERRED]
- [[test_decision_normalization_and_thresholding()]] - `calls` [INFERRED]
- [[test_generate_valid_single_decision_json_accepts_decisions_wrapper()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_applies_confidence_calibration()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_caps_confidence_with_generation_probability()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_keeps_optional_confidence_audit_fields()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_keeps_structured_trade_fields()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_requires_minimum_reward_risk_for_actionable_trade()]] - `calls` [INFERRED]
- [[test_validate_and_normalize_wraps_single_decision_object()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Decision_Engine_Tests