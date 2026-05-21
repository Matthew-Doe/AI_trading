---
type: community
cohesion: 0.09
members: 51
---

# Run Reporting Pipeline

**Cohesion:** 0.09 - loosely connected
**Members:** 51 nodes

## Members
- [[Consolidates trades, reasons (debates), and market stats for AI debugging.]] - rationale - trading_system/reporting.py
- [[Inner]] - code - tests/test_utils.py
- [[Outer]] - code - tests/test_utils.py
- [[_render_html()]] - code - trading_system/reporting.py
- [[build_config()]] - code - trading_system/main.py
- [[build_mock_order_plans()]] - code - trading_system/main.py
- [[build_run_id()]] - code - trading_system/utils.py
- [[build_run_metrics()]] - code - trading_system/main.py
- [[build_run_report_payload()]] - code - trading_system/reporting.py
- [[build_telegram_token_summary()]] - code - trading_system/main.py
- [[build_universe()]] - code - trading_system/main.py
- [[dataclass_to_dict()]] - code - trading_system/utils.py
- [[debate_result_from_dict()]] - code - trading_system/main.py
- [[ensure_dir()]] - code - trading_system/utils.py
- [[generate_report_for_latest_run()]] - code - trading_system/main.py
- [[generate_report_for_run()]] - code - trading_system/main.py
- [[get_logger()]] - code - trading_system/utils.py
- [[load_replay_data()]] - code - trading_system/main.py
- [[main()_5]] - code - trading_system/main.py
- [[main.py]] - code - trading_system/main.py
- [[manual_check()]] - code - manual_decision_check.py
- [[manual_decision_check.py]] - code - manual_decision_check.py
- [[mock_debates()]] - code - trading_system/main.py
- [[mock_decisions()]] - code - trading_system/main.py
- [[order_plan_from_dict()]] - code - trading_system/main.py
- [[parse_args()_2]] - code - trading_system/main.py
- [[print_cli_summary()]] - code - trading_system/main.py
- [[rebuild()]] - code - rebuild_debug_logs.py
- [[rebuild_debug_logs.py]] - code - rebuild_debug_logs.py
- [[reporting.py]] - code - trading_system/reporting.py
- [[run_backtest()]] - code - backtest_engine.py
- [[run_pipeline()]] - code - trading_system/main.py
- [[summarize_execution()]] - code - trading_system/main.py
- [[symbol_debate_from_dict()]] - code - trading_system/main.py
- [[symbol_market_data_from_dict()]] - code - trading_system/main.py
- [[test_dataclass_to_dict_serializes_nested_datetime_and_date()]] - code - tests/test_utils.py
- [[test_generate_report_for_latest_run()]] - code - tests/test_main_mock.py
- [[test_main_mock.py]] - code - tests/test_main_mock.py
- [[test_mock_exit_review_phase_skips_new_entry_plans()]] - code - tests/test_main_mock.py
- [[test_mock_pipeline_runs_successfully()]] - code - tests/test_main_mock.py
- [[test_reporting.py]] - code - tests/test_reporting.py
- [[test_utils.py]] - code - tests/test_utils.py
- [[test_write_run_report_creates_json_and_html()]] - code - tests/test_reporting.py
- [[test_write_run_report_writes_broker_audit_events()]] - code - tests/test_reporting.py
- [[trade_decision_from_dict()]] - code - trading_system/main.py
- [[utils.py]] - code - trading_system/utils.py
- [[write_ai_debug_log()]] - code - trading_system/reporting.py
- [[write_human_summary()]] - code - trading_system/main.py
- [[write_json()]] - code - trading_system/utils.py
- [[write_live_paper_readiness_if_accepted()]] - code - backtest_engine.py
- [[write_run_report()]] - code - trading_system/reporting.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Run_Reporting_Pipeline
SORT file.name ASC
```

## Connections to other communities
- 18 edges to [[_COMMUNITY_Market Data Loading]]
- 13 edges to [[_COMMUNITY_Broker Test Doubles]]
- 9 edges to [[_COMMUNITY_LLM Debate Handling]]
- 8 edges to [[_COMMUNITY_Live Execution Controls]]
- 7 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 6 edges to [[_COMMUNITY_Backtest Report Builders]]
- 6 edges to [[_COMMUNITY_Dashboard API Tests]]
- 3 edges to [[_COMMUNITY_Candidate Selection]]
- 3 edges to [[_COMMUNITY_Decision Engine Tests]]
- 3 edges to [[_COMMUNITY_Portfolio Summary Tests]]
- 2 edges to [[_COMMUNITY_Weekly Review Flow]]
- 1 edge to [[_COMMUNITY_Confidence Calibration Tests]]

## Top bridge nodes
- [[run_backtest()]] - degree 14, connects to 8 communities
- [[run_pipeline()]] - degree 27, connects to 6 communities
- [[write_json()]] - degree 18, connects to 4 communities
- [[utils.py]] - degree 10, connects to 3 communities
- [[ensure_dir()]] - degree 8, connects to 3 communities