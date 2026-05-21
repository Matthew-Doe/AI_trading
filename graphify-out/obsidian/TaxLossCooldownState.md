---
source_file: "trading_system/tax_state.py"
type: "code"
community: "Broker Test Doubles"
location: "L10"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Broker_Test_Doubles
---

# TaxLossCooldownState

## Connections
- [[.__init__()_13]] - `method` [EXTRACTED]
- [[.__init__()_4]] - `calls` [INFERRED]
- [[.blocked_until()]] - `method` [EXTRACTED]
- [[.is_blocked()]] - `method` [EXTRACTED]
- [[.load()]] - `method` [EXTRACTED]
- [[.record_loss_sale()]] - `method` [EXTRACTED]
- [[.save()_1]] - `method` [EXTRACTED]
- [[.snapshot()]] - `method` [EXTRACTED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[tax_state.py]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_submit_records_entry_state()]] - `calls` [INFERRED]
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - `calls` [INFERRED]
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - `calls` [INFERRED]
- [[test_submit_orders_records_live_tax_loss_cooldown()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Broker_Test_Doubles