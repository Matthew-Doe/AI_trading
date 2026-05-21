---
type: community
cohesion: 0.08
members: 74
---

# Broker Test Doubles

**Cohesion:** 0.08 - loosely connected
**Members:** 74 nodes

## Members
- [[.__init__()_2]] - code - tests/test_execution.py
- [[.__init__()_1]] - code - tests/test_execution.py
- [[.__init__()_6]] - code - trading_system/live_strategy_state.py
- [[.__init__()_13]] - code - trading_system/tax_state.py
- [[._load()]] - code - trading_system/live_strategy_state.py
- [[.blocked_until()]] - code - trading_system/tax_state.py
- [[.cancel_order_by_id()]] - code - tests/test_execution.py
- [[.clear_position()]] - code - trading_system/live_strategy_state.py
- [[.error()_2]] - code - tests/test_execution.py
- [[.get_account()_1]] - code - tests/test_execution.py
- [[.get_all_positions()_1]] - code - tests/test_execution.py
- [[.get_debate_model()]] - code - trading_system/config.py
- [[.get_decision_model()]] - code - trading_system/config.py
- [[.get_orders()]] - code - tests/test_execution.py
- [[.increment_thesis_deferral()]] - code - trading_system/live_strategy_state.py
- [[.info()_4]] - code - tests/test_execution.py
- [[.is_blocked()]] - code - trading_system/tax_state.py
- [[.load()]] - code - trading_system/tax_state.py
- [[.mark_partial_exit()]] - code - trading_system/live_strategy_state.py
- [[.reconcile_positions()]] - code - trading_system/live_strategy_state.py
- [[.record_entry()]] - code - trading_system/live_strategy_state.py
- [[.record_loss_sale()]] - code - trading_system/tax_state.py
- [[.record_tax_cooldown()]] - code - trading_system/live_strategy_state.py
- [[.record_tax_reentry()]] - code - trading_system/live_strategy_state.py
- [[.request_trade_approval()]] - code - tests/test_execution.py
- [[.save()]] - code - trading_system/live_strategy_state.py
- [[.save()_1]] - code - trading_system/tax_state.py
- [[.send_message()_1]] - code - tests/test_execution.py
- [[.send_trade_summary()]] - code - tests/test_execution.py
- [[.snapshot()]] - code - trading_system/tax_state.py
- [[.submit_order()]] - code - tests/test_execution.py
- [[.telegram_enabled()]] - code - trading_system/config.py
- [[.validate_for_live_run()]] - code - trading_system/config.py
- [[.validate_llm_provider()]] - code - trading_system/config.py
- [[.warning()_5]] - code - tests/test_execution.py
- [[DummyLogger_5]] - code - tests/test_execution.py
- [[ExecutionError]] - code - trading_system/execution.py
- [[FakeTelegramNotifier]] - code - tests/test_execution.py
- [[FakeTradingClient_1]] - code - tests/test_execution.py
- [[LivePositionState]] - code - trading_system/live_strategy_state.py
- [[LiveStrategyStateStore]] - code - trading_system/live_strategy_state.py
- [[TaxLossCooldownState]] - code - trading_system/tax_state.py
- [[TradingConfig]] - code - trading_system/config.py
- [[_now()]] - code - trading_system/live_strategy_state.py
- [[live_strategy_state.py]] - code - trading_system/live_strategy_state.py
- [[load_mock_universe()]] - code - trading_system/main.py
- [[tax_state.py]] - code - trading_system/tax_state.py
- [[test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high()]] - code - tests/test_execution.py
- [[test_daily_loss_kill_switch_blocks_new_orders()]] - code - tests/test_execution.py
- [[test_evaluate_held_positions_avoids_selling_long_at_loss()]] - code - tests/test_execution.py
- [[test_evaluate_held_positions_returns_three_signal_types()]] - code - tests/test_execution.py
- [[test_execution.py]] - code - tests/test_execution.py
- [[test_high_confidence_long_can_use_telegram_override()]] - code - tests/test_execution.py
- [[test_high_confidence_long_stays_at_standard_cap_without_telegram_approval()]] - code - tests/test_execution.py
- [[test_live_backtest_style_defers_thesis_failure_and_persists_count()]] - code - tests/test_execution.py
- [[test_live_backtest_style_held_position_plans_use_state_review()]] - code - tests/test_execution.py
- [[test_live_backtest_style_partial_exit_only_once()]] - code - tests/test_execution.py
- [[test_live_backtest_style_submit_records_entry_state()]] - code - tests/test_execution.py
- [[test_live_paper_backtest_style_bypasses_telegram_approval()]] - code - tests/test_execution.py
- [[test_live_paper_backtest_style_refuses_non_paper_url()]] - code - tests/test_execution.py
- [[test_live_paper_backtest_style_requires_empty_account_and_bias_readiness()]] - code - tests/test_execution.py
- [[test_live_strategy_state.py]] - code - tests/test_live_strategy_state.py
- [[test_live_strategy_state_persists_partial_deferral_and_tax_notes()]] - code - tests/test_live_strategy_state.py
- [[test_live_strategy_state_reconciles_missing_positions()]] - code - tests/test_live_strategy_state.py
- [[test_live_tax_adjusted_reentry_allows_reduced_size()]] - code - tests/test_execution.py
- [[test_live_tax_loss_cooldown_blocks_new_long_entries()]] - code - tests/test_execution.py
- [[test_long_keeps_standard_cap_when_available_cash_is_not_high()]] - code - tests/test_execution.py
- [[test_long_uses_cash_rich_cap_when_available_cash_is_high()]] - code - tests/test_execution.py
- [[test_order_plan_includes_limit_stop_and_take_profit_prices()]] - code - tests/test_execution.py
- [[test_review_pending_orders_cancels_large_extended_hours_move_before_open()]] - code - tests/test_execution.py
- [[test_review_pending_orders_skips_after_open()]] - code - tests/test_execution.py
- [[test_submit_orders_dry_run_includes_broker_lifecycle_fields()]] - code - tests/test_execution.py
- [[test_submit_orders_records_live_tax_loss_cooldown()]] - code - tests/test_execution.py
- [[test_submit_orders_records_skip_when_open_order_exists()]] - code - tests/test_execution.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Broker_Test_Doubles
SORT file.name ASC
```

## Connections to other communities
- 30 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 28 edges to [[_COMMUNITY_Live Execution Controls]]
- 21 edges to [[_COMMUNITY_Market Data Loading]]
- 15 edges to [[_COMMUNITY_Decision Engine Tests]]
- 13 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 9 edges to [[_COMMUNITY_Portfolio Summary Tests]]
- 8 edges to [[_COMMUNITY_Configuration Tests]]
- 8 edges to [[_COMMUNITY_LLM Debate Handling]]
- 5 edges to [[_COMMUNITY_Dashboard API Tests]]
- 4 edges to [[_COMMUNITY_Backtest Report Builders]]
- 3 edges to [[_COMMUNITY_Candidate Selection]]

## Top bridge nodes
- [[TradingConfig]] - degree 120, connects to 10 communities
- [[load_mock_universe()]] - degree 26, connects to 4 communities
- [[ExecutionError]] - degree 16, connects to 3 communities
- [[FakeTradingClient_1]] - degree 35, connects to 2 communities
- [[DummyLogger_5]] - degree 30, connects to 2 communities