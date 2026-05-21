---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L561"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_live_tax_loss_cooldown_blocks_new_long_entries()

## Connections
- [[DummyLogger_5]] - `calls` [EXTRACTED]
- [[FakeTelegramNotifier]] - `calls` [EXTRACTED]
- [[FakeTradingClient_1]] - `calls` [EXTRACTED]
- [[TaxLossCooldownState]] - `calls` [INFERRED]
- [[TradeDecision]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[load_mock_universe()]] - `calls` [INFERRED]
- [[test_execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles