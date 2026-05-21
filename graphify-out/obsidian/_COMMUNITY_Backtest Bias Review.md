---
type: community
cohesion: 0.05
members: 48
---

# Backtest Bias Review

**Cohesion:** 0.05 - loosely connected
**Members:** 48 nodes

## Members
- [[AI Trading System]] - document - README.md
- [[Acceptance Grade Bias Controls]] - rationale - docs/superpowers/plans/2026-04-30-backtest-bias-fixes.md
- [[Advanced Performance Metrics]] - code - performance_analyzer.py
- [[Analyze Backtest Correlation Report]] - code - analyze_backtest_correlation.py
- [[Backtest Bias Fixes Implementation Plan]] - document - docs/superpowers/plans/2026-04-30-backtest-bias-fixes.md
- [[Backtest Failure Skip Decision]] - code - backtest_engine.py
- [[Backtest Outcome Analysis Runner]] - code - performance_analyzer.py
- [[Bias Controls Report Metadata]] - code - backtest_engine.py
- [[Build Backtest Report]] - code - backtest_engine.py
- [[Completed Backtest Caveats]] - rationale - README.md
- [[Confidence Analysis Builder]] - code - backtest_engine.py
- [[Confidence Buckets]] - code - backtest_engine.py
- [[Confidence Tax Exit Improvements Implementation Plan]] - document - confidence_review/implementation_plan.md
- [[Confidence as Expected Value]] - rationale - confidence_review/confidence_correlation_strategies.md
- [[Exit Counterfactual Tracking]] - rationale - confidence_review/exit_logic_review.md
- [[Exit Logic Review]] - document - confidence_review/exit_logic_review.md
- [[Exploratory January Strategy Results]] - document - confidence_review/exploratory_january_results.md
- [[External Review Prompt]] - document - README.md
- [[External Technical Strategy Review Request]] - document - external_review/request.md
- [[Historical Outcome Calibration Layer]] - rationale - confidence_review/confidence_correlation_strategies.md
- [[Hold Tax Partial January Run]] - rationale - confidence_review/exploratory_january_results.md
- [[January Profit Experiments Implementation Plan]] - document - docs/superpowers/plans/2026-04-28-january-profit-experiments.md
- [[Live Paper Hold Tax Partial Implementation Plan]] - document - docs/superpowers/plans/2026-04-29-live-paper-hold-tax-partial.md
- [[Live Paper Readiness Gate]] - code - backtest_engine.py
- [[Live and Backtest Sizing Parity]] - rationale - profitability_improvement_review.md
- [[Load Symbols From Snapshot]] - code - preload_market_data.py
- [[Lot Level Tax State]] - rationale - confidence_review/tax_strategy_review.md
- [[Main Modules Overview]] - document - README.md
- [[Manual Decision Check]] - code - manual_decision_check.py
- [[Measurement Without Behavior Change]] - rationale - confidence_review/implementation_plan.md
- [[Partial Profit Taking and Scaling Out]] - rationale - confidence_review/exit_logic_review.md
- [[Phased Backtest Sequence]] - rationale - confidence_review/implementation_plan.md
- [[Preload Alpaca Daily Bars]] - code - preload_market_data.py
- [[Preload Market Data CLI]] - code - preload_market_data.py
- [[Profitability Improvement Review]] - document - profitability_improvement_review.md
- [[Raw Evidence Confidence Audit]] - rationale - confidence_review/confidence_correlation_strategies.md
- [[Rebuild AI Debug Logs]] - code - rebuild_debug_logs.py
- [[Run Backtest]] - code - backtest_engine.py
- [[Runtime Dependencies]] - document - requirements.txt
- [[Staged Exit Logic Recommendation]] - rationale - profitability_improvement_review.md
- [[Strategies to Improve Confidence Correlation]] - document - confidence_review/confidence_correlation_strategies.md
- [[Substitution Baskets]] - rationale - confidence_review/tax_strategy_review.md
- [[Tax Adjusted Expected Value]] - rationale - confidence_review/tax_strategy_review.md
- [[Tax Aware Strategy Review]] - document - confidence_review/tax_strategy_review.md
- [[Tax Blocked Shadow Trades]] - rationale - confidence_review/tax_strategy_review.md
- [[Volatility Aware Stops]] - rationale - confidence_review/exit_logic_review.md
- [[Weekly Trade Logging Implementation Plan]] - document - docs/superpowers/plans/2026-04-29-weekly-trade-logging.md
- [[Weekly Trade Review CLI]] - code - scripts/weekly_trade_review.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Backtest_Bias_Review
SORT file.name ASC
```
