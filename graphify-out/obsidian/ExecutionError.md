---
source_file: "trading_system/execution.py"
type: "code"
community: "Broker Test Doubles"
location: "L35"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Broker_Test_Doubles
---

# ExecutionError

## Connections
- [[.validate_live_paper_backtest_style_readiness()]] - `calls` [EXTRACTED]
- [[DummyLogger_5]] - `uses` [INFERRED]
- [[FakeTelegramNotifier]] - `uses` [INFERRED]
- [[FakeTradingClient_1]] - `uses` [INFERRED]
- [[HeldPositionSignal]] - `uses` [INFERRED]
- [[LivePositionState]] - `uses` [INFERRED]
- [[LiveStrategyStateStore]] - `uses` [INFERRED]
- [[OrderPlan]] - `uses` [INFERRED]
- [[PendingOrderReview]] - `uses` [INFERRED]
- [[RuntimeError]] - `inherits` [EXTRACTED]
- [[SymbolMarketData]] - `uses` [INFERRED]
- [[TaxLossCooldownState]] - `uses` [INFERRED]
- [[TelegramNotifier]] - `uses` [INFERRED]
- [[TradeDecision]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[execution.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Broker_Test_Doubles