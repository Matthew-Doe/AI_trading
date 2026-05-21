---
source_file: "trading_system/main.py"
type: "code"
community: "Broker Test Doubles"
location: "L272"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Broker_Test_Doubles
---

# load_mock_universe()

## Connections
- [[IndicatorSnapshot]] - `calls` [INFERRED]
- [[PremarketSnapshot]] - `calls` [INFERRED]
- [[SymbolMarketData]] - `calls` [INFERRED]
- [[_write_mock_run()]] - `calls` [INFERRED]
- [[build_universe()]] - `calls` [EXTRACTED]
- [[main.py]] - `contains` [EXTRACTED]
- [[test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high()]] - `calls` [INFERRED]
- [[test_candidate_selection_excludes_untradeable_symbols()]] - `calls` [INFERRED]
- [[test_candidate_selection_penalizes_overextended_symbols()]] - `calls` [INFERRED]
- [[test_candidate_selection_returns_ranked_subset()]] - `calls` [INFERRED]
- [[test_daily_loss_kill_switch_blocks_new_orders()]] - `calls` [INFERRED]
- [[test_evaluate_held_positions_avoids_selling_long_at_loss()]] - `calls` [INFERRED]
- [[test_evaluate_held_positions_returns_three_signal_types()]] - `calls` [INFERRED]
- [[test_high_confidence_long_can_use_telegram_override()]] - `calls` [INFERRED]
- [[test_high_confidence_long_stays_at_standard_cap_without_telegram_approval()]] - `calls` [INFERRED]
- [[test_live_backtest_style_held_position_plans_use_state_review()]] - `calls` [INFERRED]
- [[test_live_paper_backtest_style_bypasses_telegram_approval()]] - `calls` [INFERRED]
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - `calls` [INFERRED]
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - `calls` [INFERRED]
- [[test_load_mock_universe_includes_index_proxies()]] - `calls` [INFERRED]
- [[test_long_keeps_standard_cap_when_available_cash_is_not_high()]] - `calls` [INFERRED]
- [[test_long_uses_cash_rich_cap_when_available_cash_is_high()]] - `calls` [INFERRED]
- [[test_order_plan_includes_limit_stop_and_take_profit_prices()]] - `calls` [INFERRED]
- [[test_review_pending_orders_cancels_large_extended_hours_move_before_open()]] - `calls` [INFERRED]
- [[test_write_run_report_creates_json_and_html()]] - `calls` [INFERRED]
- [[test_write_run_report_writes_broker_audit_events()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Broker_Test_Doubles