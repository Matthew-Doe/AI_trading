---
type: community
cohesion: 0.16
members: 22
---

# Backtest Report Builders

**Cohesion:** 0.16 - loosely connected
**Members:** 22 nodes

## Members
- [[BacktestTradeRecord]] - code - trading_system/backtest_execution.py
- [[_bucket_label()]] - code - backtest_engine.py
- [[_correlation()]] - code - backtest_engine.py
- [[_mean()]] - code - backtest_engine.py
- [[_median()]] - code - backtest_engine.py
- [[_ratio()]] - code - backtest_engine.py
- [[_summarize_confidence_rows()]] - code - backtest_engine.py
- [[_trade_confidence_row()]] - code - backtest_engine.py
- [[backtest_engine.py]] - code - backtest_engine.py
- [[build_backtest_report()]] - code - backtest_engine.py
- [[build_backtest_skip_decision()]] - code - backtest_engine.py
- [[build_bias_controls()]] - code - backtest_engine.py
- [[build_confidence_analysis()]] - code - backtest_engine.py
- [[get_config_hash()]] - code - backtest_engine.py
- [[test_backtest_engine.py]] - code - tests/test_backtest_engine.py
- [[test_build_backtest_report_documents_assumptions_and_limitations()]] - code - tests/test_backtest_engine.py
- [[test_build_backtest_report_includes_bias_controls_and_acceptance_warnings()]] - code - tests/test_backtest_engine.py
- [[test_build_backtest_report_references_audit_event_file()]] - code - tests/test_backtest_engine.py
- [[test_confidence_analysis_buckets_profit_factor_median_and_correlations()]] - code - tests/test_backtest_engine.py
- [[test_confidence_analysis_handles_empty_trades()]] - code - tests/test_backtest_engine.py
- [[test_confidence_analysis_zero_variance_correlation_is_zero()]] - code - tests/test_backtest_engine.py
- [[test_live_paper_readiness_file_written_only_for_accepted_holdout()]] - code - tests/test_backtest_engine.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Backtest_Report_Builders
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 6 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 4 edges to [[_COMMUNITY_Broker Test Doubles]]
- 1 edge to [[_COMMUNITY_Market Data Loading]]
- 1 edge to [[_COMMUNITY_Live Execution Controls]]

## Top bridge nodes
- [[BacktestTradeRecord]] - degree 9, connects to 3 communities
- [[test_build_backtest_report_documents_assumptions_and_limitations()]] - degree 4, connects to 2 communities
- [[test_build_backtest_report_includes_bias_controls_and_acceptance_warnings()]] - degree 4, connects to 2 communities
- [[test_build_backtest_report_references_audit_event_file()]] - degree 4, connects to 2 communities
- [[build_backtest_skip_decision()]] - degree 3, connects to 2 communities