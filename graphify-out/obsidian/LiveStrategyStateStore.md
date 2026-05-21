---
source_file: "trading_system/live_strategy_state.py"
type: "code"
community: "Broker Test Doubles"
location: "L26"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Broker_Test_Doubles
---

# LiveStrategyStateStore

## Connections
- [[.__init__()_6]] - `method` [EXTRACTED]
- [[.__init__()_4]] - `calls` [INFERRED]
- [[._load()]] - `method` [EXTRACTED]
- [[.clear_position()]] - `method` [EXTRACTED]
- [[.increment_thesis_deferral()]] - `method` [EXTRACTED]
- [[.mark_partial_exit()]] - `method` [EXTRACTED]
- [[.reconcile_positions()]] - `method` [EXTRACTED]
- [[.record_entry()]] - `method` [EXTRACTED]
- [[.record_tax_cooldown()]] - `method` [EXTRACTED]
- [[.record_tax_reentry()]] - `method` [EXTRACTED]
- [[.save()]] - `method` [EXTRACTED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[live_strategy_state.py]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_defers_thesis_failure_and_persists_count()]] - `calls` [INFERRED]
- [[test_live_backtest_style_held_position_plans_use_state_review()]] - `calls` [INFERRED]
- [[test_live_backtest_style_partial_exit_only_once()]] - `calls` [INFERRED]
- [[test_live_backtest_style_submit_records_entry_state()]] - `calls` [INFERRED]
- [[test_live_strategy_state_persists_partial_deferral_and_tax_notes()]] - `calls` [INFERRED]
- [[test_live_strategy_state_reconciles_missing_positions()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Broker_Test_Doubles