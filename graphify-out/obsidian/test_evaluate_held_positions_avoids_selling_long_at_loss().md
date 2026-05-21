---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L68"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_evaluate_held_positions_avoids_selling_long_at_loss()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[TradeDecision]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[load_mock_universe()]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles