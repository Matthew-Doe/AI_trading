---
type: community
cohesion: 0.07
members: 38
---

# Crosscutting Test Contracts

**Cohesion:** 0.07 - loosely connected
**Members:** 38 nodes

## Members
- [[Alpaca Execution Engine Contract]] - code - tests/test_execution.py
- [[Backtest Execution Engine Contract]] - code - tests/test_backtest_execution.py
- [[Backtest Report Contract]] - code - tests/test_backtest_engine.py
- [[Behavior Experiment Exits]] - code - tests/test_backtest_execution.py
- [[Bias Safe Fill Modes]] - code - tests/test_backtest_execution.py
- [[Bias Safe Live Paper Workflow]] - rationale - tests/test_backtest_engine.py
- [[Candidate Selection Contract]] - code - tests/test_selection.py
- [[Confidence Analysis Contract]] - code - tests/test_backtest_engine.py
- [[Confidence Calibration Contract]] - code - tests/test_confidence_calibration.py
- [[Confidence and Sizing Controls]] - rationale - tests/test_backtest_execution.py
- [[Dashboard Payload Contract]] - code - tests/test_dashboard.py
- [[Data Quality Contract]] - code - tests/test_data.py
- [[Dataclass Serialization Contract]] - code - tests/test_utils.py
- [[Debate JSON Extraction Contract]] - code - tests/test_debate.py
- [[Decision Normalization Contract]] - code - tests/test_decision.py
- [[Historical Decision Outcome Builder]] - code - tests/test_confidence_calibration.py
- [[Live Paper Daily Gate]] - code - tests/test_run_live_paper_daily.py
- [[Live Paper Phase Schedule]] - code - tests/test_scheduler.py
- [[Live Paper Readiness Gate_1]] - code - tests/test_backtest_engine.py
- [[Live Position Review Contract]] - code - tests/test_execution.py
- [[Live Strategy State Persistence Contract]] - code - tests/test_live_strategy_state.py
- [[Live Tax Controls Contract]] - code - tests/test_execution.py
- [[Market Close Summary Contract]] - code - tests/test_portfolio_summary.py
- [[Market Data Universe Contract]] - code - tests/test_data.py
- [[Mock Pipeline Contract]] - code - tests/test_main_mock.py
- [[Performance Payload Contract]] - code - tests/test_dashboard.py
- [[Point In Time Universe Contract]] - code - tests/test_config.py
- [[Report Audit Dashboard Artifacts]] - rationale - tests/test_dashboard.py
- [[Run Report Contract]] - code - tests/test_reporting.py
- [[Single Symbol Decision Generation]] - code - tests/test_decision.py
- [[Tax Loss Cooldown Controls]] - rationale - tests/test_backtest_execution.py
- [[Tax Shadow Accounting]] - code - tests/test_backtest_execution.py
- [[Telegram Trade Message Contract]] - code - tests/test_telegram.py
- [[Token Usage Contract]] - code - tests/test_llm.py
- [[Trade Audit Contract]] - code - tests/test_trade_audit.py
- [[Trading Config Environment Contract]] - code - tests/test_config.py
- [[Trading System Package Marker]] - code - trading_system/__init__.py
- [[Weekly Review Contract]] - code - tests/test_weekly_review.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Crosscutting_Test_Contracts
SORT file.name ASC
```
