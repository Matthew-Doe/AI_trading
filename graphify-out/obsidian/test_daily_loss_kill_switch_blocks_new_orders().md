---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L431"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_daily_loss_kill_switch_blocks_new_orders()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[TradeDecision]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[load_mock_universe()]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles