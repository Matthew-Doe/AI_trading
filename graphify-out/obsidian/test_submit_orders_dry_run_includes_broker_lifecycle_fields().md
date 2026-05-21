---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L487"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_submit_orders_dry_run_includes_broker_lifecycle_fields()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[OrderPlan]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles