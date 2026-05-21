---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L630"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_submit_orders_records_live_tax_loss_cooldown()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[OrderPlan]] - `calls` [INFERRED]
- [[TaxLossCooldownState]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles