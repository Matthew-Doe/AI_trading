---
type: community
cohesion: 0.12
members: 33
---

# Confidence Calibration Tests

**Cohesion:** 0.12 - loosely connected
**Members:** 33 nodes

## Members
- [[.__init__()_17]] - code - trading_system/confidence_calibration.py
- [[.__init__()_9]] - code - trading_system/decision.py
- [[._ensure_loaded()]] - code - trading_system/confidence_calibration.py
- [[.calibrate()_2]] - code - trading_system/confidence_calibration.py
- [[.matches()]] - code - trading_system/confidence_calibration.py
- [[ConfidenceBucket]] - code - trading_system/confidence_calibration.py
- [[ConfidenceCalibrator]] - code - trading_system/confidence_calibration.py
- [[HistoricalDecisionOutcome]] - code - trading_system/confidence_calibration.py
- [[_build_confidence_buckets()]] - code - trading_system/confidence_calibration.py
- [[_parse_known_timestamp()]] - code - trading_system/confidence_calibration.py
- [[_parse_run_started_at()]] - code - trading_system/confidence_calibration.py
- [[build_historical_decision_outcomes()]] - code - trading_system/confidence_calibration.py
- [[calibrate_confidence()]] - code - trading_system/confidence_calibration.py
- [[confidence_calibration.py]] - code - trading_system/confidence_calibration.py
- [[label_decision_correctness()]] - code - trading_system/confidence_calibration.py
- [[test_build_historical_decision_outcomes_excludes_forward_outcome_not_yet_known()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_includes_forward_outcome_after_known_time()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_reads_backtest_day_folders()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_returns_labeled_outcome()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_skips_current_backtest_day()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_skips_future_run_ids_without_lookup()]] - code - tests/test_confidence_calibration.py
- [[test_build_historical_decision_outcomes_skips_incomplete_forward_window()]] - code - tests/test_decision.py
- [[test_build_historical_decision_outcomes_skips_runs_without_forward_history()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_excludes_non_terminal_upper_bound()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_falls_back_when_bucket_is_sparse()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_includes_terminal_upper_bound()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_returns_raw_when_unmatched()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_uses_bucket_hit_rate()]] - code - tests/test_confidence_calibration.py
- [[test_calibrate_confidence_uses_bucket_hit_rate_at_exact_minimum_samples()]] - code - tests/test_confidence_calibration.py
- [[test_confidence_calibration.py]] - code - tests/test_confidence_calibration.py
- [[test_confidence_calibrator_returns_raw_when_calibration_disabled()]] - code - tests/test_confidence_calibration.py
- [[test_label_decision_correctness_for_long_short_and_skip()]] - code - tests/test_confidence_calibration.py
- [[test_label_decision_correctness_rejects_invalid_action()]] - code - tests/test_confidence_calibration.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Confidence_Calibration_Tests
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_Market Data Loading]]
- 4 edges to [[_COMMUNITY_Decision Engine Tests]]
- 1 edge to [[_COMMUNITY_Run Reporting Pipeline]]
- 1 edge to [[_COMMUNITY_LLM Debate Handling]]

## Top bridge nodes
- [[ConfidenceCalibrator]] - degree 10, connects to 3 communities
- [[.__init__()_9]] - degree 4, connects to 3 communities
- [[build_historical_decision_outcomes()]] - degree 15, connects to 1 community
- [[ConfidenceBucket]] - degree 9, connects to 1 community
- [[test_confidence_calibrator_returns_raw_when_calibration_disabled()]] - degree 3, connects to 1 community