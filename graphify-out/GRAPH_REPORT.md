# Graph Report - .  (2026-05-21)

## Corpus Check
- Large corpus: 213 files · ~526,977 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 758 nodes · 1828 edges · 25 communities (23 shown, 2 thin omitted)
- Extraction: 64% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 648 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Backtest Execution Tests|Backtest Execution Tests]]
- [[_COMMUNITY_Broker Test Doubles|Broker Test Doubles]]
- [[_COMMUNITY_Market Data Loading|Market Data Loading]]
- [[_COMMUNITY_Run Reporting Pipeline|Run Reporting Pipeline]]
- [[_COMMUNITY_Live Execution Controls|Live Execution Controls]]
- [[_COMMUNITY_LLM Debate Handling|LLM Debate Handling]]
- [[_COMMUNITY_Backtest Bias Review|Backtest Bias Review]]
- [[_COMMUNITY_Dashboard API Tests|Dashboard API Tests]]
- [[_COMMUNITY_Decision Engine Tests|Decision Engine Tests]]
- [[_COMMUNITY_Crosscutting Test Contracts|Crosscutting Test Contracts]]
- [[_COMMUNITY_Trading System Core|Trading System Core]]
- [[_COMMUNITY_Confidence Calibration Tests|Confidence Calibration Tests]]
- [[_COMMUNITY_Portfolio Summary Tests|Portfolio Summary Tests]]
- [[_COMMUNITY_Configuration Tests|Configuration Tests]]
- [[_COMMUNITY_Backtest Report Builders|Backtest Report Builders]]
- [[_COMMUNITY_Weekly Review Flow|Weekly Review Flow]]
- [[_COMMUNITY_Candidate Selection|Candidate Selection]]
- [[_COMMUNITY_Daily Runner CLI|Daily Runner CLI]]
- [[_COMMUNITY_Performance Analyzer|Performance Analyzer]]
- [[_COMMUNITY_Daily Rollout Plan|Daily Rollout Plan]]
- [[_COMMUNITY_Preload Cache Contract|Preload Cache Contract]]
- [[_COMMUNITY_Package Metadata|Package Metadata]]

## God Nodes (most connected - your core abstractions)
1. `TradingConfig` - 120 edges
2. `BacktestExecutionEngine` - 60 edges
3. `TradeDecision` - 55 edges
4. `MarketDataService` - 53 edges
5. `FakeTradingClient` - 35 edges
6. `DecisionEngine` - 34 edges
7. `AlpacaExecutionEngine` - 31 edges
8. `DummyLogger` - 30 edges
9. `FakeTelegramNotifier` - 28 edges
10. `OrderPlan` - 27 edges

## Surprising Connections (you probably didn't know these)
- `Historical Outcome Calibration Layer` --shares_data_with--> `Confidence Analysis Builder`  [INFERRED]
  confidence_review/confidence_correlation_strategies.md → backtest_engine.py
- `Backtest Bias Fixes Implementation Plan` --conceptually_related_to--> `Load Symbols From Snapshot`  [INFERRED]
  docs/superpowers/plans/2026-04-30-backtest-bias-fixes.md → preload_market_data.py
- `Rebuild AI Debug Logs` --conceptually_related_to--> `Weekly Trade Logging Implementation Plan`  [INFERRED]
  rebuild_debug_logs.py → docs/superpowers/plans/2026-04-29-weekly-trade-logging.md
- `Staged Exit Logic Recommendation` --semantically_similar_to--> `Exit Logic Review`  [INFERRED] [semantically similar]
  profitability_improvement_review.md → confidence_review/exit_logic_review.md
- `test_load_mock_universe_includes_index_proxies()` --calls--> `load_mock_universe()`  [INFERRED]
  tests/test_data.py → trading_system/main.py

## Hyperedges (group relationships)
- **Confidence Tax Exit Review Tracks** — confidence_correlation_strategies_review, tax_strategy_review, exit_logic_review, confidence_implementation_plan [EXTRACTED 1.00]
- **Hold Tax Partial Strategy Bundle** — exit_logic_partial_profit, tax_strategy_tax_adjusted_ev, exploratory_hold_tax_partial_run, live_paper_hold_tax_partial_plan [EXTRACTED 1.00]
- **Bias Safe Backtest Acceptance Bundle** — backtest_bias_fixes_plan, backtest_engine_bias_controls, backtest_engine_live_paper_readiness, exploratory_january_results [INFERRED 0.80]
- **Backtest Live Parity Controls** — test_backtest_engine_live_paper_readiness_gate, test_execution_live_position_review_contract, test_scheduler_live_paper_phase_schedule, crosscutting_bias_safe_live_paper [INFERRED 0.80]
- **Tax Cooldown and Shadow Accounting** — test_backtest_execution_tax_shadow_accounting, test_execution_live_tax_controls_contract, test_live_strategy_state_persistence_contract, crosscutting_tax_loss_cooldown [INFERRED 0.85]
- **Run Artifact Observability** — test_reporting_run_report_contract, test_dashboard_dashboard_payload_contract, test_trade_audit_trade_audit_contract, crosscutting_report_audit_dashboard [INFERRED 0.85]
- **Daily Trading Pipeline** — data_build_universe, selection_CandidateSelector, debate_OllamaDebateEngine, decision_DecisionEngine, execution_AlpacaExecutionEngine, reporting_write_run_report [EXTRACTED 1.00]
- **Tax-Loss Cooldown System** — execution_AlpacaExecutionEngine, tax_state_TaxLossCooldownState, live_strategy_state_LiveStrategyStateStore, backtest_execution_tax_shadow [INFERRED 0.85]
- **Run Observability Outputs** — main_run_artifacts, reporting_write_run_report, dashboard_build_dashboard_payload, weekly_review_build_weekly_review, verify_fix_verify [INFERRED 0.80]

## Communities (25 total, 2 thin omitted)

### Community 0 - "Backtest Execution Tests"
Cohesion: 0.06
Nodes (52): _symbol_data(), test_backtest_execution_blocks_rebuy_after_loss_for_wash_sale_cooldown(), test_backtest_execution_long_entry_and_exit(), test_backtest_execution_short_stop_loss(), test_backtest_execution_uses_simulated_open_for_new_entries(), test_backtest_tax_summary_estimates_wash_sale_rebuy_loss_deferral(), test_bias_safe_previous_close_entry_ignores_open_print(), test_conditional_hold_extension_defers_thesis_failed_exit() (+44 more)

### Community 1 - "Broker Test Doubles"
Cohesion: 0.08
Nodes (35): DummyLogger, FakeTelegramNotifier, FakeTradingClient, test_buy_more_signal_uses_cash_rich_cap_when_available_cash_is_high(), test_daily_loss_kill_switch_blocks_new_orders(), test_evaluate_held_positions_avoids_selling_long_at_loss(), test_evaluate_held_positions_returns_three_signal_types(), test_high_confidence_long_can_use_telegram_override() (+27 more)

### Community 2 - "Market Data Loading"
Cohesion: 0.08
Nodes (32): load_symbols_from_snapshot(), main(), parse_args(), _parse_utc_date(), preload_alpaca_daily_bars(), DummyLogger, _symbol_with_metrics(), test_alpha_vantage_daily_bars_parse_ohlcv_response() (+24 more)

### Community 3 - "Run Reporting Pipeline"
Cohesion: 0.09
Nodes (43): run_backtest(), write_live_paper_readiness_if_accepted(), manual_check(), rebuild(), test_generate_report_for_latest_run(), test_mock_exit_review_phase_skips_new_entry_plans(), test_mock_pipeline_runs_successfully(), test_write_run_report_creates_json_and_html() (+35 more)

### Community 4 - "Live Execution Controls"
Cohesion: 0.08
Nodes (20): RuntimeError, DummyLogger, test_humanize_trade_reason_for_sizing_trace(), test_humanize_trade_reason_preserves_plain_language_reason(), test_send_message_is_best_effort_when_request_fails(), AlpacaExecutionEngine, _broker_lifecycle_payload(), _empty_broker_lifecycle_payload() (+12 more)

### Community 5 - "LLM Debate Handling"
Cohesion: 0.08
Nodes (22): DummyLogger, test_token_usage_tracker_aggregates_calls(), test_token_usage_tracker_keeps_logprob_fields(), _build_json_repair_prompt(), DebateError, _extract_json(), _llm_generate(), OllamaDebateEngine (+14 more)

### Community 6 - "Backtest Bias Review"
Cohesion: 0.05
Nodes (48): Analyze Backtest Correlation Report, Acceptance Grade Bias Controls, Backtest Bias Fixes Implementation Plan, Bias Controls Report Metadata, Build Backtest Report, Confidence Analysis Builder, Confidence Buckets, Live Paper Readiness Gate (+40 more)

### Community 7 - "Dashboard API Tests"
Cohesion: 0.09
Nodes (34): BaseHTTPRequestHandler, test_benchmark_history_from_frame_handles_yfinance_multiindex_columns(), test_build_dashboard_payload_includes_latest_run_and_log_tail(), test_build_dashboard_payload_uses_newest_log_for_in_progress_run(), test_build_performance_payload_normalizes_portfolio_against_benchmark(), test_dashboard_handler_rejects_unknown_api_path(), test_dashboard_handler_serves_latest_api_json(), test_dashboard_handler_serves_performance_api() (+26 more)

### Community 8 - "Decision Engine Tests"
Cohesion: 0.12
Nodes (27): _dummy_debate(), DummyLogger, IdentityCalibrator, StubCalibrator, test_decide_per_symbol_skips_symbol_after_repeated_json_failures(), test_decide_uses_single_symbol_generation_without_batch_probe(), test_decision_normalization_and_thresholding(), test_generate_valid_single_decision_json_accepts_decisions_wrapper() (+19 more)

### Community 9 - "Crosscutting Test Contracts"
Cohesion: 0.07
Nodes (38): Bias Safe Live Paper Workflow, Confidence and Sizing Controls, Report Audit Dashboard Artifacts, Tax Loss Cooldown Controls, Backtest Report Contract, Confidence Analysis Contract, Live Paper Readiness Gate, Backtest Execution Engine Contract (+30 more)

### Community 10 - "Trading System Core"
Cohesion: 0.08
Nodes (37): BacktestExecutionEngine, Exit Counterfactuals, Tax Shadow Positions, ConfidenceCalibrator, Historical Decision Outcomes, TradingConfig, build_dashboard_payload, Performance Payload (+29 more)

### Community 11 - "Confidence Calibration Tests"
Cohesion: 0.12
Nodes (26): test_build_historical_decision_outcomes_excludes_forward_outcome_not_yet_known(), test_build_historical_decision_outcomes_includes_forward_outcome_after_known_time(), test_build_historical_decision_outcomes_reads_backtest_day_folders(), test_build_historical_decision_outcomes_returns_labeled_outcome(), test_build_historical_decision_outcomes_skips_current_backtest_day(), test_build_historical_decision_outcomes_skips_future_run_ids_without_lookup(), test_build_historical_decision_outcomes_skips_runs_without_forward_history(), test_calibrate_confidence_excludes_non_terminal_upper_bound() (+18 more)

### Community 12 - "Portfolio Summary Tests"
Cohesion: 0.11
Nodes (14): DummyLogger, FakeMarketData, FakeTelegram, FakeTradingClient, test_build_market_close_summary_message_uses_relative_performance(), test_market_close_reporter_sends_telegram_message(), test_live_paper_phase_schedule_uses_entry_and_exit_times(), build_market_close_summary_message() (+6 more)

### Community 13 - "Configuration Tests"
Cohesion: 0.11
Nodes (17): DummyLogger, _fresh_trading_config(), test_behavior_experiment_flags_default_disabled(), test_behavior_experiment_flags_read_from_environment(), test_bias_safe_backtest_config_defaults(), test_bias_safe_backtest_config_reads_environment(), test_bias_safe_historical_premarket_snapshot_does_not_use_full_day_volume(), test_fetch_forward_close_window_uses_next_three_trading_days() (+9 more)

### Community 14 - "Backtest Report Builders"
Cohesion: 0.16
Nodes (20): _bucket_label(), build_backtest_report(), build_backtest_skip_decision(), build_bias_controls(), build_confidence_analysis(), _correlation(), get_config_hash(), _mean() (+12 more)

### Community 15 - "Weekly Review Flow"
Cohesion: 0.22
Nodes (16): main(), _report(), test_weekly_review_flags_adaptation_candidates_and_renders_markdown(), test_weekly_review_summarizes_performance_breakdowns_and_contributions(), test_write_weekly_review_outputs_json_and_markdown(), _adaptation_candidates(), _breakdown(), build_weekly_review() (+8 more)

### Community 16 - "Candidate Selection"
Cohesion: 0.26
Nodes (6): DummyLogger, test_candidate_selection_excludes_untradeable_symbols(), test_candidate_selection_penalizes_overextended_symbols(), test_candidate_selection_returns_ranked_subset(), CandidateSelector, clamp()

### Community 18 - "Daily Runner CLI"
Cohesion: 0.70
Nodes (4): main(), _main_args(), should_run(), _start_date()

### Community 19 - "Performance Analyzer"
Cohesion: 0.67
Nodes (3): calculate_metrics(), Calculates advanced performance and risk metrics.     trades: list of trade dict, run_analysis()

### Community 20 - "Daily Rollout Plan"
Cohesion: 0.67
Nodes (4): Live Paper Daily Rollout Design, Live Paper Daily Rollout Implementation Plan, Live Paper Start Date Guard, Live Paper Daily Runner

## Ambiguous Edges - Review These
- `Mock Pipeline Contract` → `Trading System Package Marker`  [AMBIGUOUS]
  trading_system/__init__.py · relation: conceptually_related_to

## Knowledge Gaps
- **35 isolated node(s):** `RunArtifacts`, `Main Modules Overview`, `External Review Prompt`, `Confidence Buckets`, `Backtest Failure Skip Decision` (+30 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Mock Pipeline Contract` and `Trading System Package Marker`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `TradingConfig` connect `Broker Test Doubles` to `Backtest Execution Tests`, `Market Data Loading`, `Run Reporting Pipeline`, `Live Execution Controls`, `LLM Debate Handling`, `Dashboard API Tests`, `Decision Engine Tests`, `Portfolio Summary Tests`, `Configuration Tests`, `Backtest Report Builders`?**
  _High betweenness centrality (0.282) - this node is a cross-community bridge._
- **Why does `BacktestExecutionEngine` connect `Backtest Execution Tests` to `Market Data Loading`, `Run Reporting Pipeline`, `Live Execution Controls`, `Backtest Report Builders`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `MarketDataService` connect `Market Data Loading` to `Broker Test Doubles`, `Run Reporting Pipeline`, `LLM Debate Handling`, `Decision Engine Tests`, `Confidence Calibration Tests`, `Portfolio Summary Tests`, `Configuration Tests`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 114 inferred relationships involving `TradingConfig` (e.g. with `DummyLogger` and `FakeTradingClient`) actually correct?**
  _`TradingConfig` has 114 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `BacktestExecutionEngine` (e.g. with `OrderPlan` and `TradeDecision`) actually correct?**
  _`BacktestExecutionEngine` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `TradeDecision` (e.g. with `DummyLogger` and `FakeTradingClient`) actually correct?**
  _`TradeDecision` has 54 INFERRED edges - model-reasoned connections that need verification._