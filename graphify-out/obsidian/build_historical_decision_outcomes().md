---
source_file: "trading_system/confidence_calibration.py"
type: "code"
community: "Confidence Calibration Tests"
location: "L84"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Confidence_Calibration_Tests
---

# build_historical_decision_outcomes()

## Connections
- [[._ensure_loaded()]] - `calls` [EXTRACTED]
- [[HistoricalDecisionOutcome]] - `calls` [EXTRACTED]
- [[_parse_known_timestamp()]] - `calls` [EXTRACTED]
- [[_parse_run_started_at()]] - `calls` [EXTRACTED]
- [[confidence_calibration.py]] - `contains` [EXTRACTED]
- [[label_decision_correctness()]] - `calls` [EXTRACTED]
- [[read_json()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_excludes_forward_outcome_not_yet_known()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_includes_forward_outcome_after_known_time()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_reads_backtest_day_folders()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_returns_labeled_outcome()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_skips_current_backtest_day()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_skips_future_run_ids_without_lookup()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_skips_incomplete_forward_window()]] - `calls` [INFERRED]
- [[test_build_historical_decision_outcomes_skips_runs_without_forward_history()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Confidence_Calibration_Tests