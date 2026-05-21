---
type: community
cohesion: 0.12
members: 44
---

# Decision Engine Tests

**Cohesion:** 0.12 - loosely connected
**Members:** 44 nodes

## Members
- [[.__init__()_3]] - code - tests/test_decision.py
- [[._build_decision_prompt()]] - code - trading_system/decision.py
- [[._build_single_symbol_prompt()]] - code - trading_system/decision.py
- [[._calibrate_confidence()]] - code - trading_system/decision.py
- [[._confidence_cap_reason()]] - code - trading_system/decision.py
- [[._decide_per_symbol()]] - code - trading_system/decision.py
- [[._generate_valid_decision_json()]] - code - trading_system/decision.py
- [[._generate_valid_single_decision_json()]] - code - trading_system/decision.py
- [[._validate_and_normalize()]] - code - trading_system/decision.py
- [[.calibrate()]] - code - tests/test_decision.py
- [[.calibrate()_1]] - code - tests/test_decision.py
- [[.decide()]] - code - trading_system/decision.py
- [[.info()_5]] - code - tests/test_decision.py
- [[.warning()_6]] - code - tests/test_decision.py
- [[DecisionEngine]] - code - trading_system/decision.py
- [[DecisionError]] - code - trading_system/decision.py
- [[DummyLogger_6]] - code - tests/test_decision.py
- [[IdentityCalibrator]] - code - tests/test_decision.py
- [[StubCalibrator]] - code - tests/test_decision.py
- [[_build_json_repair_prompt()]] - code - trading_system/decision.py
- [[_build_single_symbol_json_repair_prompt()]] - code - trading_system/decision.py
- [[_decision_confidence_cap()]] - code - trading_system/decision.py
- [[_dummy_debate()]] - code - tests/test_decision.py
- [[_extract_json()]] - code - trading_system/decision.py
- [[_normalize_decision_payload()]] - code - trading_system/decision.py
- [[_normalize_single_decision_payload()]] - code - trading_system/decision.py
- [[_ollama_generate()]] - code - trading_system/decision.py
- [[_optional_float()]] - code - trading_system/decision.py
- [[_optional_int()]] - code - trading_system/decision.py
- [[_optional_text()]] - code - trading_system/decision.py
- [[_reward_risk_ratio()]] - code - trading_system/decision.py
- [[decision.py]] - code - trading_system/decision.py
- [[test_decide_per_symbol_skips_symbol_after_repeated_json_failures()]] - code - tests/test_decision.py
- [[test_decide_uses_single_symbol_generation_without_batch_probe()]] - code - tests/test_decision.py
- [[test_decision.py]] - code - tests/test_decision.py
- [[test_decision_normalization_and_thresholding()]] - code - tests/test_decision.py
- [[test_extract_json_handles_preamble_and_trailing_text()]] - code - tests/test_decision.py
- [[test_generate_valid_single_decision_json_accepts_decisions_wrapper()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_applies_confidence_calibration()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_caps_confidence_with_generation_probability()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_keeps_optional_confidence_audit_fields()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_keeps_structured_trade_fields()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_requires_minimum_reward_risk_for_actionable_trade()]] - code - tests/test_decision.py
- [[test_validate_and_normalize_wraps_single_decision_object()]] - code - tests/test_decision.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Decision_Engine_Tests
SORT file.name ASC
```

## Connections to other communities
- 15 edges to [[_COMMUNITY_Broker Test Doubles]]
- 6 edges to [[_COMMUNITY_LLM Debate Handling]]
- 5 edges to [[_COMMUNITY_Market Data Loading]]
- 4 edges to [[_COMMUNITY_Confidence Calibration Tests]]
- 4 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 4 edges to [[_COMMUNITY_Candidate Selection]]
- 3 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 1 edge to [[_COMMUNITY_Live Execution Controls]]

## Top bridge nodes
- [[DecisionEngine]] - degree 34, connects to 6 communities
- [[DecisionError]] - degree 19, connects to 6 communities
- [[DummyLogger_6]] - degree 17, connects to 2 communities
- [[IdentityCalibrator]] - degree 15, connects to 2 communities
- [[._validate_and_normalize()]] - degree 12, connects to 2 communities