---
source_file: "trading_system/models.py"
type: "code"
community: "Live Execution Controls"
location: "L91"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Live_Execution_Controls
---

# OrderPlan

## Connections
- [[.build_held_position_order_plans()]] - `calls` [INFERRED]
- [[.build_order_plans()]] - `calls` [INFERRED]
- [[.review_live_backtest_style_position()]] - `calls` [INFERRED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[BacktestExecutionEngine]] - `uses` [INFERRED]
- [[BacktestPosition]] - `uses` [INFERRED]
- [[BacktestTradeRecord]] - `uses` [INFERRED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[DummyLogger_8]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[ExitCounterfactualRecord]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[PendingExitCounterfactual]] - `uses` [INFERRED]
- [[TaxShadowPosition]] - `uses` [INFERRED]
- [[TaxShadowTradeRecord]] - `uses` [INFERRED]
- [[TelegramApprovalDecision]] - `uses` [INFERRED]
- [[TelegramError]] - `uses` [INFERRED]
- [[TelegramNotifier]] - `uses` [INFERRED]
- [[build_mock_order_plans()]] - `calls` [INFERRED]
- [[models.py]] - `contains` [EXTRACTED]
- [[order_plan_from_dict()]] - `calls` [INFERRED]
- [[test_humanize_trade_reason_for_sizing_trace()]] - `calls` [INFERRED]
- [[test_humanize_trade_reason_preserves_plain_language_reason()]] - `calls` [INFERRED]
- [[test_live_backtest_style_submit_records_entry_state()]] - `calls` [INFERRED]
- [[test_submit_orders_dry_run_includes_broker_lifecycle_fields()]] - `calls` [INFERRED]
- [[test_submit_orders_records_live_tax_loss_cooldown()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Live_Execution_Controls