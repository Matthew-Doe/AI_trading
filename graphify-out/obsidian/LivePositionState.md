---
source_file: "trading_system/live_strategy_state.py"
type: "code"
community: "Broker Test Doubles"
location: "L12"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Broker_Test_Doubles
---

# LivePositionState

## Connections
- [[._load()]] - `calls` [EXTRACTED]
- [[.submit_orders()]] - `calls` [INFERRED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[live_strategy_state.py]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_defers_thesis_failure_and_persists_count()]] - `calls` [INFERRED]
- [[test_live_backtest_style_held_position_plans_use_state_review()]] - `calls` [INFERRED]
- [[test_live_backtest_style_partial_exit_only_once()]] - `calls` [INFERRED]
- [[test_live_strategy_state_persists_partial_deferral_and_tax_notes()]] - `calls` [INFERRED]
- [[test_live_strategy_state_reconciles_missing_positions()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Broker_Test_Doubles