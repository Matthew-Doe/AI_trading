---
source_file: "tests/test_execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Broker_Test_Doubles
---

# test_execution.py

## Connections
- [[DummyLogger_5]] - `contains` [EXTRACTED]
- [[FakeTelegramNotifier]] - `contains` [EXTRACTED]
- [[FakeTradingClient_1]] - `contains` [EXTRACTED]
- [[test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high()]] - `contains` [EXTRACTED]
- [[test_daily_loss_kill_switch_blocks_new_orders()]] - `contains` [EXTRACTED]
- [[test_evaluate_held_positions_avoids_selling_long_at_loss()]] - `contains` [EXTRACTED]
- [[test_evaluate_held_positions_returns_three_signal_types()]] - `contains` [EXTRACTED]
- [[test_high_confidence_long_can_use_telegram_override()]] - `contains` [EXTRACTED]
- [[test_high_confidence_long_stays_at_standard_cap_without_telegram_approval()]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_defers_thesis_failure_and_persists_count()]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_held_position_plans_use_state_review()]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_partial_exit_only_once()]] - `contains` [EXTRACTED]
- [[test_live_backtest_style_submit_records_entry_state()]] - `contains` [EXTRACTED]
- [[test_live_paper_backtest_style_bypasses_telegram_approval()]] - `contains` [EXTRACTED]
- [[test_live_paper_backtest_style_refuses_non_paper_url()]] - `contains` [EXTRACTED]
- [[test_live_paper_backtest_style_requires_empty_account_and_bias_readiness()]] - `contains` [EXTRACTED]
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - `contains` [EXTRACTED]
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - `contains` [EXTRACTED]
- [[test_long_keeps_standard_cap_when_available_cash_is_not_high()]] - `contains` [EXTRACTED]
- [[test_long_uses_cash_rich_cap_when_available_cash_is_high()]] - `contains` [EXTRACTED]
- [[test_order_plan_includes_limit_stop_and_take_profit_prices()]] - `contains` [EXTRACTED]
- [[test_review_pending_orders_cancels_large_extended_hours_move_before_open()]] - `contains` [EXTRACTED]
- [[test_review_pending_orders_skips_after_open()]] - `contains` [EXTRACTED]
- [[test_submit_orders_dry_run_includes_broker_lifecycle_fields()]] - `contains` [EXTRACTED]
- [[test_submit_orders_records_live_tax_loss_cooldown()]] - `contains` [EXTRACTED]
- [[test_submit_orders_records_skip_when_open_order_exists()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Broker_Test_Doubles