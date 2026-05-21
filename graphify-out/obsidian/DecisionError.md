---
source_file: "trading_system/decision.py"
type: "code"
community: "Decision Engine Tests"
location: "L16"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Decision_Engine_Tests
---

# DecisionError

## Connections
- [[._generate_valid_decision_json()]] - `calls` [EXTRACTED]
- [[._generate_valid_single_decision_json()]] - `calls` [EXTRACTED]
- [[._validate_and_normalize()]] - `calls` [EXTRACTED]
- [[ConfidenceCalibrator]] - `uses` [INFERRED]
- [[DummyLogger_6]] - `uses` [INFERRED]
- [[IdentityCalibrator]] - `uses` [INFERRED]
- [[LLMClient]] - `uses` [INFERRED]
- [[MarketDataService]] - `uses` [INFERRED]
- [[RuntimeError]] - `inherits` [EXTRACTED]
- [[StubCalibrator]] - `uses` [INFERRED]
- [[SymbolDebate]] - `uses` [INFERRED]
- [[TokenUsageTracker]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[_extract_json()]] - `calls` [EXTRACTED]
- [[_normalize_decision_payload()]] - `calls` [EXTRACTED]
- [[_normalize_single_decision_payload()]] - `calls` [EXTRACTED]
- [[decision.py]] - `contains` [EXTRACTED]
- [[test_decide_per_symbol_skips_symbol_after_repeated_json_failures()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Decision_Engine_Tests