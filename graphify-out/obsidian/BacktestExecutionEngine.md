---
source_file: "trading_system/backtest_execution.py"
type: "code"
community: "Backtest Execution Tests"
location: "L150"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Backtest_Execution_Tests
---

# BacktestExecutionEngine

## Connections
- [[.__init__()_11]] - `method` [EXTRACTED]
- [[._append_audit_event()]] - `method` [EXTRACTED]
- [[._apply_liquidity_cap()]] - `method` [EXTRACTED]
- [[._close_position()]] - `method` [EXTRACTED]
- [[._close_tax_shadow_position()]] - `method` [EXTRACTED]
- [[._confidence_size_multiplier()]] - `method` [EXTRACTED]
- [[._conservative_daily_path_exit()]] - `method` [EXTRACTED]
- [[._defer_thesis_failure_exit()]] - `method` [EXTRACTED]
- [[._delayed_exit_net_pnl()]] - `method` [EXTRACTED]
- [[._entry_fill_price()]] - `method` [EXTRACTED]
- [[._entry_price()]] - `method` [EXTRACTED]
- [[._future_replacement_buy_matches()]] - `method` [EXTRACTED]
- [[._is_tax_blocked()]] - `method` [EXTRACTED]
- [[._open_tax_shadow_position()]] - `method` [EXTRACTED]
- [[._planned_exit_prices()]] - `method` [EXTRACTED]
- [[._process_exit_counterfactuals()]] - `method` [EXTRACTED]
- [[._process_tax_shadow_positions()]] - `method` [EXTRACTED]
- [[._size_position()]] - `method` [EXTRACTED]
- [[._staged_exit_decision()]] - `method` [EXTRACTED]
- [[._start_exit_counterfactual()]] - `method` [EXTRACTED]
- [[._summarize_exit_counterfactual_group()]] - `method` [EXTRACTED]
- [[._take_partial_profit()]] - `method` [EXTRACTED]
- [[._tax_adjusted_reentry_decision()_1]] - `method` [EXTRACTED]
- [[.get_exit_counterfactuals_analysis()]] - `method` [EXTRACTED]
- [[.get_summary()]] - `method` [EXTRACTED]
- [[.get_tax_shadow_analysis()]] - `method` [EXTRACTED]
- [[.get_tax_summary()]] - `method` [EXTRACTED]
- [[.process_decisions()]] - `method` [EXTRACTED]
- [[.update_equity()]] - `method` [EXTRACTED]
- [[AuditEvent]] - `uses` [INFERRED]
- [[OrderPlan]] - `uses` [INFERRED]
- [[SymbolMarketData]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[backtest_execution.py]] - `contains` [EXTRACTED]
- [[run_backtest()]] - `calls` [INFERRED]
- [[test_backtest_execution_blocks_rebuy_after_loss_for_wash_sale_cooldown()]] - `calls` [INFERRED]
- [[test_backtest_execution_long_entry_and_exit()]] - `calls` [INFERRED]
- [[test_backtest_execution_short_stop_loss()]] - `calls` [INFERRED]
- [[test_backtest_execution_uses_simulated_open_for_new_entries()]] - `calls` [INFERRED]
- [[test_backtest_tax_summary_estimates_wash_sale_rebuy_loss_deferral()]] - `calls` [INFERRED]
- [[test_backtest_writes_entry_partial_and_full_audit_events()]] - `calls` [INFERRED]
- [[test_bias_safe_previous_close_entry_ignores_open_print()]] - `calls` [INFERRED]
- [[test_build_backtest_report_documents_assumptions_and_limitations()]] - `calls` [INFERRED]
- [[test_build_backtest_report_includes_bias_controls_and_acceptance_warnings()]] - `calls` [INFERRED]
- [[test_build_backtest_report_references_audit_event_file()]] - `calls` [INFERRED]
- [[test_conditional_hold_extension_defers_thesis_failed_exit()]] - `calls` [INFERRED]
- [[test_conditional_hold_extension_exits_when_adverse_move_is_too_large()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_halves_mid_confidence_size()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_skips_below_floor_confidence()]] - `calls` [INFERRED]
- [[test_confidence_sizing_experiment_uses_full_size_for_high_confidence()]] - `calls` [INFERRED]
- [[test_conservative_daily_high_low_exit_chooses_stop_when_stop_and_target_touched()]] - `calls` [INFERRED]
- [[test_exit_counterfactuals_track_delayed_long_pl_after_stop_loss()]] - `calls` [INFERRED]
- [[test_exit_counterfactuals_track_delayed_short_pl_and_group_by_exit_reason()]] - `calls` [INFERRED]
- [[test_open_plus_delay_fill_uses_delayed_price_not_decision_open()]] - `calls` [INFERRED]
- [[test_partial_profit_taking_closes_fraction_and_keeps_residual_position()]] - `calls` [INFERRED]
- [[test_partial_profit_taking_residual_closes_on_later_stop()]] - `calls` [INFERRED]
- [[test_realistic_friction_caps_quantity_to_liquidity_and_records_cost()]] - `calls` [INFERRED]
- [[test_tax_adjusted_reentry_allows_high_ev_downsized_trade()]] - `calls` [INFERRED]
- [[test_tax_adjusted_reentry_still_blocks_low_ev_trade()]] - `calls` [INFERRED]
- [[test_tax_shadow_trade_closes_without_affecting_real_portfolio_state()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/Backtest_Execution_Tests