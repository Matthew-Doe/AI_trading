---
source_file: "trading_system/execution.py"
type: "code"
community: "Live Execution Controls"
location: "L39"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Live_Execution_Controls
---

# AlpacaExecutionEngine

## Connections
- [[.__init__()_4]] - `method` [EXTRACTED]
- [[._build_order_request()]] - `method` [EXTRACTED]
- [[._daily_loss_limit_reached()]] - `method` [EXTRACTED]
- [[._is_before_market_open()]] - `method` [EXTRACTED]
- [[._is_tax_loss_blocked()]] - `method` [EXTRACTED]
- [[._planned_prices()]] - `method` [EXTRACTED]
- [[._standard_trade_cap()]] - `method` [EXTRACTED]
- [[._tax_adjusted_reentry_decision()]] - `method` [EXTRACTED]
- [[.build_held_position_order_plans()]] - `method` [EXTRACTED]
- [[.build_order_plans()]] - `method` [EXTRACTED]
- [[.evaluate_held_positions()]] - `method` [EXTRACTED]
- [[.get_tax_state_snapshot()]] - `method` [EXTRACTED]
- [[.review_live_backtest_style_position()]] - `method` [EXTRACTED]
- [[.review_pending_orders()]] - `method` [EXTRACTED]
- [[.submit_orders()]] - `method` [EXTRACTED]
- [[.validate_live_paper_backtest_style_readiness()]] - `method` [EXTRACTED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[HeldPositionSignal]] - `uses` [INFERRED]
- [[LivePositionState]] - `uses` [INFERRED]
- [[LiveStrategyStateStore]] - `uses` [INFERRED]
- [[OrderPlan]] - `uses` [INFERRED]
- [[PendingOrderReview]] - `uses` [INFERRED]
- [[SymbolMarketData]] - `uses` [INFERRED]
- [[TaxLossCooldownState]] - `uses` [INFERRED]
- [[TelegramNotifier]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[execution.py]] - `contains` [EXTRACTED]
- [[run_pipeline()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/Live_Execution_Controls