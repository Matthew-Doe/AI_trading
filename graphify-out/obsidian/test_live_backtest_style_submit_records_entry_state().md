---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L665"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_live_backtest_style_submit_records_entry_state()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[LiveStrategyStateStore]] - `calls` [INFERRED]
- [[OrderPlan]] - `calls` [INFERRED]
- [[TaxLossCooldownState]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles