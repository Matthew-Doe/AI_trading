---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L402"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_order_plan_includes_limit_stop_and_take_profit_prices()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[TradeDecision]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[load_mock_universe()]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles