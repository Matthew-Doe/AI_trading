---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L89"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# FakeTelegramNotifier

## Connections
- [[.__init__()_2]] - `method` [EXTRACTED]
- [[.request_trade_approval()]] - `method` [EXTRACTED]
- [[.send_message()_1]] - `method` [EXTRACTED]
- [[.send_trade_summary()]] - `method` [EXTRACTED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[LivePositionState]] - `uses` [INFERRED]
- [[LiveStrategyStateStore]] - `uses` [INFERRED]
- [[OrderPlan]] - `uses` [INFERRED]
- [[TaxLossCooldownState]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[test_daily_loss_kill_switch_blocks_new_orders()]] - `calls` [EXTRACTED]
- [[test_execution.py]] - `contains` [EXTRACTED]
- [[test_high_confidence_long_can_use_telegram_override()]] - `calls` [EXTRACTED]
- [[test_high_confidence_long_stays_at_standard_cap_without_telegram_approval()]] - `calls` [EXTRACTED]
- [[test_live_backtest_style_submit_records_entry_state()]] - `calls` [EXTRACTED]
- [[test_live_paper_backtest_style_bypasses_telegram_approval()]] - `calls` [EXTRACTED]
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - `calls` [EXTRACTED]
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - `calls` [EXTRACTED]
- [[test_long_keeps_standard_cap_when_available_cash_is_not_high()]] - `calls` [EXTRACTED]
- [[test_long_uses_cash_rich_cap_when_available_cash_is_high()]] - `calls` [EXTRACTED]
- [[test_order_plan_includes_limit_stop_and_take_profit_prices()]] - `calls` [EXTRACTED]
- [[test_review_pending_orders_cancels_large_extended_hours_move_before_open()]] - `calls` [EXTRACTED]
- [[test_review_pending_orders_skips_after_open()]] - `calls` [EXTRACTED]
- [[test_submit_orders_dry_run_includes_broker_lifecycle_fields()]] - `calls` [EXTRACTED]
- [[test_submit_orders_records_live_tax_loss_cooldown()]] - `calls` [EXTRACTED]
- [[test_submit_orders_records_skip_when_open_order_exists()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles