---
type: community
cohesion: 0.08
members: 37
---

# Trading System Core

**Cohesion:** 0.08 - loosely connected
**Members:** 37 nodes

## Members
- [[AlpacaExecutionEngine_1]] - code - trading_system/execution.py
- [[AuditEvent_1]] - code - trading_system/trade_audit.py
- [[BacktestExecutionEngine_1]] - code - trading_system/backtest_execution.py
- [[CandidateSelector_1]] - code - trading_system/selection.py
- [[ConfidenceCalibrator_1]] - code - trading_system/confidence_calibration.py
- [[DecisionEngine_1]] - code - trading_system/decision.py
- [[Exit Counterfactuals]] - code - trading_system/backtest_execution.py
- [[HeldPositionSignal_1]] - code - trading_system/models.py
- [[Historical Decision Outcomes]] - code - trading_system/confidence_calibration.py
- [[JSON and Dataclass Utilities]] - code - trading_system/utils.py
- [[LLMClient_1]] - code - trading_system/llm.py
- [[Live Paper Backtest-Style Execution]] - code - trading_system/execution.py
- [[LiveStrategyStateStore_1]] - code - trading_system/live_strategy_state.py
- [[MarketCloseReporter_1]] - code - trading_system/portfolio_summary.py
- [[MarketDataService_1]] - code - trading_system/data.py
- [[MarketDataService.build_universe]] - code - trading_system/data.py
- [[MarketDataService.fetch_symbol_market_data]] - code - trading_system/data.py
- [[OllamaDebateEngine_1]] - code - trading_system/debate.py
- [[OrderPlan_1]] - code - trading_system/models.py
- [[PendingOrderReview_1]] - code - trading_system/models.py
- [[Per-Run Artifact Files]] - code - trading_system/main.py
- [[Performance Payload]] - code - trading_system/dashboard.py
- [[SymbolDebate_1]] - code - trading_system/models.py
- [[SymbolMarketData_1]] - code - trading_system/models.py
- [[Tax Shadow Positions]] - code - trading_system/backtest_execution.py
- [[TaxLossCooldownState_1]] - code - trading_system/tax_state.py
- [[TelegramNotifier_1]] - code - trading_system/telegram.py
- [[TokenUsageTracker_1]] - code - trading_system/llm.py
- [[TradeDecision_1]] - code - trading_system/models.py
- [[TradingConfig_1]] - code - trading_system/config.py
- [[Weekly Adaptation Candidates]] - code - trading_system/weekly_review.py
- [[build_dashboard_payload]] - code - trading_system/dashboard.py
- [[build_weekly_review]] - code - trading_system/weekly_review.py
- [[run_pipeline]] - code - trading_system/main.py
- [[scheduler.main]] - code - trading_system/scheduler.py
- [[verify_fix.verify]] - code - verify_fix.py
- [[write_run_report]] - code - trading_system/reporting.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Trading_System_Core
SORT file.name ASC
```
