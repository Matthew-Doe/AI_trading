---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L356"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_review_pending_orders_cancels_large_extended_hours_move_before_open()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[load_mock_universe()]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles