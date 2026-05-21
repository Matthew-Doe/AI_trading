---
type: community
cohesion: 0.09
members: 44
---

# Dashboard API Tests

**Cohesion:** 0.09 - loosely connected
**Members:** 44 nodes

## Members
- [[.__init__()_14]] - code - trading_system/dashboard.py
- [[._route_json()]] - code - trading_system/dashboard.py
- [[._send_artifact()]] - code - trading_system/dashboard.py
- [[._send_json()]] - code - trading_system/dashboard.py
- [[._send_text()]] - code - trading_system/dashboard.py
- [[.do_GET()]] - code - trading_system/dashboard.py
- [[.log_message()]] - code - trading_system/dashboard.py
- [[.performance_payload()]] - code - trading_system/dashboard.py
- [[BaseHTTPRequestHandler]] - code
- [[DashboardRequestHandler]] - code - trading_system/dashboard.py
- [[DashboardServer]] - code - trading_system/dashboard.py
- [[Returns (alpaca_period, timeframe, start_timestamp)]] - rationale - trading_system/dashboard.py
- [[ThreadingHTTPServer]] - code
- [[_build_summary_from_artifacts()]] - code - trading_system/dashboard.py
- [[_data_quality_warnings()]] - code - trading_system/dashboard.py
- [[_read_optional_json()]] - code - trading_system/dashboard.py
- [[_write_mock_run()]] - code - tests/test_dashboard.py
- [[benchmark_history_from_frame()]] - code - trading_system/dashboard.py
- [[build_dashboard_payload()]] - code - trading_system/dashboard.py
- [[build_performance_payload_from_history()]] - code - trading_system/dashboard.py
- [[dashboard.py]] - code - trading_system/dashboard.py
- [[fetch_alpaca_portfolio_history()]] - code - trading_system/dashboard.py
- [[fetch_benchmark_history()]] - code - trading_system/dashboard.py
- [[fetch_benchmark_history_alpaca()]] - code - trading_system/dashboard.py
- [[fetch_performance_payload()]] - code - trading_system/dashboard.py
- [[find_latest_log_run_id()]] - code - trading_system/dashboard.py
- [[find_latest_run_id()]] - code - trading_system/dashboard.py
- [[get_period_params()]] - code - trading_system/dashboard.py
- [[list_run_ids()]] - code - trading_system/dashboard.py
- [[load_run_payload()]] - code - trading_system/dashboard.py
- [[main()_4]] - code - trading_system/dashboard.py
- [[parse_args()_1]] - code - trading_system/dashboard.py
- [[portfolio_history_date_bounds()]] - code - trading_system/dashboard.py
- [[read_log_tail()]] - code - trading_system/dashboard.py
- [[render_dashboard_html()]] - code - trading_system/dashboard.py
- [[test_benchmark_history_from_frame_handles_yfinance_multiindex_columns()]] - code - tests/test_dashboard.py
- [[test_build_dashboard_payload_includes_latest_run_and_log_tail()]] - code - tests/test_dashboard.py
- [[test_build_dashboard_payload_uses_newest_log_for_in_progress_run()]] - code - tests/test_dashboard.py
- [[test_build_performance_payload_normalizes_portfolio_against_benchmark()]] - code - tests/test_dashboard.py
- [[test_dashboard.py]] - code - tests/test_dashboard.py
- [[test_dashboard_handler_rejects_unknown_api_path()]] - code - tests/test_dashboard.py
- [[test_dashboard_handler_serves_latest_api_json()]] - code - tests/test_dashboard.py
- [[test_dashboard_handler_serves_performance_api()]] - code - tests/test_dashboard.py
- [[test_find_latest_run_id_ignores_incomplete_runs()]] - code - tests/test_dashboard.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Dashboard_API_Tests
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 5 edges to [[_COMMUNITY_Broker Test Doubles]]
- 2 edges to [[_COMMUNITY_Market Data Loading]]

## Top bridge nodes
- [[_write_mock_run()]] - degree 14, connects to 2 communities
- [[DashboardRequestHandler]] - degree 9, connects to 1 community
- [[DashboardServer]] - degree 6, connects to 1 community
- [[load_run_payload()]] - degree 6, connects to 1 community
- [[_read_optional_json()]] - degree 5, connects to 1 community