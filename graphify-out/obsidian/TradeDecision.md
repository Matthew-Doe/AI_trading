---
source_file: "trading_system/models.py"
type: "code"
community: "Backtest Execution Tests"
location: "L68"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Backtest_Execution_Tests
---

# TradeDecision

## Connections
- [[._decide_per_symbol()]] - `calls` [INFERRED]
- [[._validate_and_normalize()]] - `calls` [INFERRED]
- [[AlpacaExecutionEngine]] - `uses` [INFERRED]
- [[AuditEvent]] - `uses` [INFERRED]
- [[BacktestExecutionEngine]] - `uses` [INFERRED]
- [[BacktestPosition]] - `uses` [INFERRED]
- [[BacktestTradeRecord]] - `uses` [INFERRED]
- [[DecisionEngine]] - `uses` [INFERRED]
- [[DecisionError]] - `uses` [INFERRED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[ExecutionError]] - `uses` [INFERRED]
- [[ExitCounterfactualRecord]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[PendingExitCounterfactual]] - `uses` [INFERRED]
- [[TaxShadowPosition]] - `uses` [INFERRED]
- [[TaxShadowTradeRecord]] - `uses` [INFERRED]
- [[build_backtest_skip_decision()]] - `calls` [INFERRED]
- [[mock_decisions()]] - `calls` [INFERRED]
- [[models.py]] - `contains` [EXTRACTED]
- [[test_backtest_execution_blocks_rebuy_after_loss_for_wash_sale_cooldown()]] - `calls` [INFERRED]
- [[test_backtest_execution_long_entry_and_exit()]] - `calls` [INFERRED]
- [[test_backtest_execution_short_stop_loss()]] - `calls` [INFERRED]
- [[test_backtest_execution_uses_simulated_open_for_new_entries()]] - `calls` [INFERRED]
- [[test_backtest_tax_summary_estimates_wash_sale_rebuy_loss_deferral()]] - `calls` [INFERRED]
- [[test_backtest_writes_entry_partial_and_full_audit_events()]] - `calls` [INFERRED]
- [[test_bias_safe_previous_close_entry_ignores_open_print()]] - `calls` [INFERRED]
- [[test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high()]] - `calls` [INFERRED]
- [[test_conditional_hold_extension_defers_thesis_failed_exit()]] - `calls` [INFERRED]
- [[test_conditional_hold_extension_exits_when_adverse_move_is_too_large()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_halves_mid_confidence_size()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_skips_below_floor_confidence()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_uses_full_size_for_high_confidence()]] - `calls` [INFERRED]
- [[test_daily_loss_kill_switch_blocks_new_orders()]] - `calls` [INFERRED]
- [[test_evaluate_held_positions_avoids_selling_long_at_loss()]] - `calls` [INFERRED]
- [[test_evaluate_held_positions_returns_three_signal_types()]] - `calls` [INFERRED]
- [[test_exit_counterfactuals_track_delayed_long_pl_after_stop_loss()]] - `calls` [INFERRED]
- [[test_exit_counterfactuals_track_delayed_short_pl_and_group_by_exit_reason()]] - `calls` [INFERRED]
- [[test_high_confidence_long_can_use_telegram_override()]] - `calls` [INFERRED]
- [[test_high_confidence_long_stays_at_standard_cap_without_telegram_approval()]] - `calls` [INFERRED]
- [[test_live_paper_backtest_style_bypasses_telegram_approval()]] - `calls` [INFERRED]
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - `calls` [INFERRED]
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - `calls` [INFERRED]
- [[test_long_keeps_standard_cap_when_available_cash_is_not_high()]] - `calls` [INFERRED]
- [[test_long_uses_cash_rich_cap_when_available_cash_is_high()]] - `calls` [INFERRED]
- [[test_open_plus_delay_fill_uses_delayed_price_not_decision_open()]] - `calls` [INFERRED]
- [[test_order_plan_includes_limit_stop_and_take_profit_prices()]] - `calls` [INFERRED]
- [[test_partial_profit_taking_closes_fraction_and_keeps_residual_position()]] - `calls` [INFERRED]
- [[test_partial_profit_taking_residual_closes_on_later_stop()]] - `calls` [INFERRED]
- [[test_realistic_friction_caps_quantity_to_liquidity_and_records_cost()]] - `calls` [INFERRED]
- [[test_stable_ids_are_repeatable_and_distinguish_inputs()]] - `calls` [INFERRED]
- [[test_tax_adjusted_reentry_allows_high_ev_downsized_trade()]] - `calls` [INFERRED]
- [[test_tax_adjusted_reentry_still_blocks_low_ev_trade()]] - `calls` [INFERRED]
- [[test_tax_shadow_trade_closes_without_affecting_real_portfolio_state()]] - `calls` [INFERRED]
- [[trade_decision_from_dict()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Backtest_Execution_Tests