---
source_file: "trading_system/main.py"
type: "code"
community: "Run Reporting Pipeline"
location: "L85"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Run_Reporting_Pipeline
---

# run_pipeline()

## Connections
- [[AlpacaExecutionEngine]] - `calls` [INFERRED]
- [[CandidateSelector]] - `calls` [INFERRED]
- [[DecisionEngine]] - `calls` [INFERRED]
- [[MarketDataService]] - `calls` [INFERRED]
- [[OllamaDebateEngine]] - `calls` [INFERRED]
- [[TelegramNotifier]] - `calls` [INFERRED]
- [[TokenUsageTracker]] - `calls` [INFERRED]
- [[build_mock_order_plans()]] - `calls` [EXTRACTED]
- [[build_run_id()]] - `calls` [INFERRED]
- [[build_run_metrics()]] - `calls` [EXTRACTED]
- [[build_telegram_token_summary()]] - `calls` [EXTRACTED]
- [[build_universe()]] - `calls` [EXTRACTED]
- [[ensure_dir()]] - `calls` [INFERRED]
- [[get_logger()]] - `calls` [INFERRED]
- [[load_replay_data()]] - `calls` [EXTRACTED]
- [[main()_5]] - `calls` [EXTRACTED]
- [[main()_3]] - `calls` [INFERRED]
- [[main.py]] - `contains` [EXTRACTED]
- [[mock_debates()]] - `calls` [EXTRACTED]
- [[mock_decisions()]] - `calls` [EXTRACTED]
- [[print_cli_summary()]] - `calls` [EXTRACTED]
- [[test_generate_report_for_latest_run()]] - `calls` [INFERRED]
- [[test_mock_exit_review_phase_skips_new_entry_plans()]] - `calls` [INFERRED]
- [[test_mock_pipeline_runs_successfully()]] - `calls` [INFERRED]
- [[write_human_summary()]] - `calls` [EXTRACTED]
- [[write_json()]] - `calls` [INFERRED]
- [[write_run_report()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Run_Reporting_Pipeline