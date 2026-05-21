---
type: community
cohesion: 0.06
members: 85
---

# Backtest Execution Tests

**Cohesion:** 0.06 - loosely connected
**Members:** 85 nodes

## Members
- [[.__init__()_11]] - code - trading_system/backtest_execution.py
- [[._append_audit_event()]] - code - trading_system/backtest_execution.py
- [[._apply_liquidity_cap()]] - code - trading_system/backtest_execution.py
- [[._close_position()]] - code - trading_system/backtest_execution.py
- [[._close_tax_shadow_position()]] - code - trading_system/backtest_execution.py
- [[._confidence_size_multiplier()]] - code - trading_system/backtest_execution.py
- [[._conservative_daily_path_exit()]] - code - trading_system/backtest_execution.py
- [[._defer_thesis_failure_exit()]] - code - trading_system/backtest_execution.py
- [[._delayed_exit_net_pnl()]] - code - trading_system/backtest_execution.py
- [[._entry_fill_price()]] - code - trading_system/backtest_execution.py
- [[._entry_price()]] - code - trading_system/backtest_execution.py
- [[._future_replacement_buy_matches()]] - code - trading_system/backtest_execution.py
- [[._is_tax_blocked()]] - code - trading_system/backtest_execution.py
- [[._open_tax_shadow_position()]] - code - trading_system/backtest_execution.py
- [[._planned_exit_prices()]] - code - trading_system/backtest_execution.py
- [[._process_exit_counterfactuals()]] - code - trading_system/backtest_execution.py
- [[._process_tax_shadow_positions()]] - code - trading_system/backtest_execution.py
- [[._size_position()]] - code - trading_system/backtest_execution.py
- [[._staged_exit_decision()]] - code - trading_system/backtest_execution.py
- [[._start_exit_counterfactual()]] - code - trading_system/backtest_execution.py
- [[._summarize_exit_counterfactual_group()]] - code - trading_system/backtest_execution.py
- [[._take_partial_profit()]] - code - trading_system/backtest_execution.py
- [[._tax_adjusted_reentry_decision()_1]] - code - trading_system/backtest_execution.py
- [[.get_exit_counterfactuals_analysis()]] - code - trading_system/backtest_execution.py
- [[.get_summary()]] - code - trading_system/backtest_execution.py
- [[.get_tax_shadow_analysis()]] - code - trading_system/backtest_execution.py
- [[.get_tax_summary()]] - code - trading_system/backtest_execution.py
- [[.process_decisions()]] - code - trading_system/backtest_execution.py
- [[.update_equity()]] - code - trading_system/backtest_execution.py
- [[AuditEvent]] - code - trading_system/trade_audit.py
- [[BacktestExecutionEngine]] - code - trading_system/backtest_execution.py
- [[BacktestPosition]] - code - trading_system/backtest_execution.py
- [[Calculates current portfolio value based on latest prices.]] - rationale - trading_system/backtest_execution.py
- [[Executes new decisions and checks existing positions for stopstargets.]] - rationale - trading_system/backtest_execution.py
- [[ExitCounterfactualRecord]] - code - trading_system/backtest_execution.py
- [[PendingExitCounterfactual]] - code - trading_system/backtest_execution.py
- [[TaxShadowPosition]] - code - trading_system/backtest_execution.py
- [[TaxShadowTradeRecord]] - code - trading_system/backtest_execution.py
- [[TradeDecision]] - code - trading_system/models.py
- [[_decision_expected_value_pct()]] - code - trading_system/backtest_execution.py
- [[_decision_snapshot()]] - code - trading_system/backtest_execution.py
- [[_mae_pct()]] - code - trading_system/backtest_execution.py
- [[_mark_price()]] - code - trading_system/backtest_execution.py
- [[_market_snapshot()]] - code - trading_system/backtest_execution.py
- [[_mfe_pct()]] - code - trading_system/backtest_execution.py
- [[_regime_snapshot()]] - code - trading_system/backtest_execution.py
- [[_return_pct()]] - code - trading_system/backtest_execution.py
- [[_risk_normalized_return()]] - code - trading_system/backtest_execution.py
- [[_symbol_data()]] - code - tests/test_backtest_execution.py
- [[_unrealized_pct()]] - code - trading_system/backtest_execution.py
- [[_update_excursions()]] - code - trading_system/backtest_execution.py
- [[append_audit_event()]] - code - trading_system/trade_audit.py
- [[backtest_execution.py]] - code - trading_system/backtest_execution.py
- [[load_audit_events()]] - code - trading_system/trade_audit.py
- [[stable_decision_id()]] - code - trading_system/trade_audit.py
- [[stable_event_id()]] - code - trading_system/trade_audit.py
- [[stable_trade_id()]] - code - trading_system/trade_audit.py
- [[test_audit_event_jsonl_round_trip()]] - code - tests/test_trade_audit.py
- [[test_backtest_execution.py]] - code - tests/test_backtest_execution.py
- [[test_backtest_execution_blocks_rebuy_after_loss_for_wash_sale_cooldown()]] - code - tests/test_backtest_execution.py
- [[test_backtest_execution_long_entry_and_exit()]] - code - tests/test_backtest_execution.py
- [[test_backtest_execution_short_stop_loss()]] - code - tests/test_backtest_execution.py
- [[test_backtest_execution_uses_simulated_open_for_new_entries()]] - code - tests/test_backtest_execution.py
- [[test_backtest_tax_summary_estimates_wash_sale_rebuy_loss_deferral()]] - code - tests/test_backtest_execution.py
- [[test_backtest_writes_entry_partial_and_full_audit_events()]] - code - tests/test_trade_audit.py
- [[test_bias_safe_previous_close_entry_ignores_open_print()]] - code - tests/test_backtest_execution.py
- [[test_conditional_hold_extension_defers_thesis_failed_exit()]] - code - tests/test_backtest_execution.py
- [[test_conditional_hold_extension_exits_when_adverse_move_is_too_large()]] - code - tests/test_backtest_execution.py
- [[test_confidence_sizing_experiment_halves_mid_confidence_size()]] - code - tests/test_backtest_execution.py
- [[test_confidence_sizing_experiment_skips_below_floor_confidence()]] - code - tests/test_backtest_execution.py
- [[test_confidence_sizing_experiment_uses_full_size_for_high_confidence()]] - code - tests/test_backtest_execution.py
- [[test_conservative_daily_high_low_exit_chooses_stop_when_stop_and_target_touched()]] - code - tests/test_backtest_execution.py
- [[test_exit_counterfactuals_track_delayed_long_pl_after_stop_loss()]] - code - tests/test_backtest_execution.py
- [[test_exit_counterfactuals_track_delayed_short_pl_and_group_by_exit_reason()]] - code - tests/test_backtest_execution.py
- [[test_open_plus_delay_fill_uses_delayed_price_not_decision_open()]] - code - tests/test_backtest_execution.py
- [[test_partial_profit_taking_closes_fraction_and_keeps_residual_position()]] - code - tests/test_backtest_execution.py
- [[test_partial_profit_taking_residual_closes_on_later_stop()]] - code - tests/test_backtest_execution.py
- [[test_realistic_friction_caps_quantity_to_liquidity_and_records_cost()]] - code - tests/test_backtest_execution.py
- [[test_stable_ids_are_repeatable_and_distinguish_inputs()]] - code - tests/test_trade_audit.py
- [[test_tax_adjusted_reentry_allows_high_ev_downsized_trade()]] - code - tests/test_backtest_execution.py
- [[test_tax_adjusted_reentry_still_blocks_low_ev_trade()]] - code - tests/test_backtest_execution.py
- [[test_tax_shadow_trade_closes_without_affecting_real_portfolio_state()]] - code - tests/test_backtest_execution.py
- [[test_trade_audit.py]] - code - tests/test_trade_audit.py
- [[trade_audit.py]] - code - trading_system/trade_audit.py
- [[write_broker_audit_events()]] - code - trading_system/reporting.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Backtest_Execution_Tests
SORT file.name ASC
```

## Connections to other communities
- 30 edges to [[_COMMUNITY_Broker Test Doubles]]
- 12 edges to [[_COMMUNITY_Market Data Loading]]
- 9 edges to [[_COMMUNITY_Backtest Report Builders]]
- 7 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 7 edges to [[_COMMUNITY_Live Execution Controls]]
- 4 edges to [[_COMMUNITY_Decision Engine Tests]]
- 1 edge to [[_COMMUNITY_LLM Debate Handling]]

## Top bridge nodes
- [[TradeDecision]] - degree 55, connects to 6 communities
- [[BacktestExecutionEngine]] - degree 60, connects to 4 communities
- [[BacktestPosition]] - degree 7, connects to 2 communities
- [[ExitCounterfactualRecord]] - degree 6, connects to 2 communities
- [[PendingExitCounterfactual]] - degree 6, connects to 2 communities