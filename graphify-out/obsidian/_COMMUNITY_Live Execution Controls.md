---
type: community
cohesion: 0.08
members: 50
---

# Live Execution Controls

**Cohesion:** 0.08 - loosely connected
**Members:** 50 nodes

## Members
- [[.__init__()_4]] - code - trading_system/execution.py
- [[.__init__()_15]] - code - trading_system/telegram.py
- [[._answer_callback_query()]] - code - trading_system/telegram.py
- [[._api_url()]] - code - trading_system/telegram.py
- [[._build_order_request()]] - code - trading_system/execution.py
- [[._daily_loss_limit_reached()]] - code - trading_system/execution.py
- [[._edit_message_reply_markup()]] - code - trading_system/telegram.py
- [[._get_updates()]] - code - trading_system/telegram.py
- [[._is_before_market_open()]] - code - trading_system/execution.py
- [[._is_tax_loss_blocked()]] - code - trading_system/execution.py
- [[._latest_update_offset()]] - code - trading_system/telegram.py
- [[._planned_prices()]] - code - trading_system/execution.py
- [[._standard_trade_cap()]] - code - trading_system/execution.py
- [[._tax_adjusted_reentry_decision()]] - code - trading_system/execution.py
- [[.build_held_position_order_plans()]] - code - trading_system/execution.py
- [[.build_order_plans()]] - code - trading_system/execution.py
- [[.evaluate_held_positions()]] - code - trading_system/execution.py
- [[.get_tax_state_snapshot()]] - code - trading_system/execution.py
- [[.is_enabled()_1]] - code - trading_system/telegram.py
- [[.request_trade_approval()_1]] - code - trading_system/telegram.py
- [[.review_live_backtest_style_position()]] - code - trading_system/execution.py
- [[.review_pending_orders()]] - code - trading_system/execution.py
- [[.send_message()_2]] - code - trading_system/telegram.py
- [[.send_trade_summary()_1]] - code - trading_system/telegram.py
- [[.submit_orders()]] - code - trading_system/execution.py
- [[.validate_live_paper_backtest_style_readiness()]] - code - trading_system/execution.py
- [[.warning()_7]] - code - tests/test_telegram.py
- [[AlpacaExecutionEngine]] - code - trading_system/execution.py
- [[DummyLogger_8]] - code - tests/test_telegram.py
- [[HeldPositionSignal]] - code - trading_system/models.py
- [[OrderPlan]] - code - trading_system/models.py
- [[PendingOrderReview]] - code - trading_system/models.py
- [[RuntimeError]] - code
- [[TelegramApprovalDecision]] - code - trading_system/telegram.py
- [[TelegramError]] - code - trading_system/telegram.py
- [[TelegramNotifier]] - code - trading_system/telegram.py
- [[_broker_lifecycle_payload()]] - code - trading_system/execution.py
- [[_empty_broker_lifecycle_payload()]] - code - trading_system/execution.py
- [[_humanize_trade_reason()]] - code - trading_system/telegram.py
- [[_parse_sizing_reason()]] - code - trading_system/telegram.py
- [[_position_unrealized_loss()]] - code - trading_system/execution.py
- [[_would_sell_long_at_loss()]] - code - trading_system/execution.py
- [[execution.py]] - code - trading_system/execution.py
- [[held_position_signal_from_dict()]] - code - trading_system/main.py
- [[pending_order_review_from_dict()]] - code - trading_system/main.py
- [[telegram.py]] - code - trading_system/telegram.py
- [[test_humanize_trade_reason_for_sizing_trace()]] - code - tests/test_telegram.py
- [[test_humanize_trade_reason_preserves_plain_language_reason()]] - code - tests/test_telegram.py
- [[test_send_message_is_best_effort_when_request_fails()]] - code - tests/test_telegram.py
- [[test_telegram.py]] - code - tests/test_telegram.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Live_Execution_Controls
SORT file.name ASC
```

## Connections to other communities
- 28 edges to [[_COMMUNITY_Broker Test Doubles]]
- 8 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 7 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 5 edges to [[_COMMUNITY_LLM Debate Handling]]
- 3 edges to [[_COMMUNITY_Market Data Loading]]
- 3 edges to [[_COMMUNITY_Portfolio Summary Tests]]
- 1 edge to [[_COMMUNITY_Decision Engine Tests]]
- 1 edge to [[_COMMUNITY_Backtest Report Builders]]

## Top bridge nodes
- [[OrderPlan]] - degree 27, connects to 5 communities
- [[AlpacaExecutionEngine]] - degree 31, connects to 4 communities
- [[RuntimeError]] - degree 6, connects to 4 communities
- [[TelegramNotifier]] - degree 22, connects to 3 communities
- [[HeldPositionSignal]] - degree 5, connects to 2 communities