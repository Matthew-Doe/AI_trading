---
type: community
cohesion: 0.11
members: 26
---

# Configuration Tests

**Cohesion:** 0.11 - loosely connected
**Members:** 26 nodes

## Members
- [[.error()_1]] - code - tests/test_config.py
- [[.info()_3]] - code - tests/test_config.py
- [[.warning()_4]] - code - tests/test_config.py
- [[DummyLogger_4]] - code - tests/test_config.py
- [[_fresh_trading_config()]] - code - tests/test_config.py
- [[_parse_bool_env()]] - code - trading_system/config.py
- [[_parse_schedule_times()]] - code - trading_system/config.py
- [[_parse_symbol_list()]] - code - trading_system/config.py
- [[_parse_time_env()]] - code - trading_system/config.py
- [[config.py]] - code - trading_system/config.py
- [[load_dotenv()]] - code - trading_system/config.py
- [[test_behavior_experiment_flags_default_disabled()]] - code - tests/test_config.py
- [[test_behavior_experiment_flags_read_from_environment()]] - code - tests/test_config.py
- [[test_bias_safe_backtest_config_defaults()]] - code - tests/test_config.py
- [[test_bias_safe_backtest_config_reads_environment()]] - code - tests/test_config.py
- [[test_bias_safe_historical_premarket_snapshot_does_not_use_full_day_volume()]] - code - tests/test_config.py
- [[test_config.py]] - code - tests/test_config.py
- [[test_fetch_forward_close_window_uses_next_three_trading_days()]] - code - tests/test_config.py
- [[test_parse_schedule_times_multiple_entries()]] - code - tests/test_config.py
- [[test_point_in_time_universe_snapshot_does_not_truncate_snapshot_symbols()]] - code - tests/test_config.py
- [[test_point_in_time_universe_snapshot_fails_closed_without_snapshot()]] - code - tests/test_config.py
- [[test_point_in_time_universe_snapshot_selects_latest_on_or_before_date()]] - code - tests/test_config.py
- [[test_trading_config_accepts_explicit_schedule_times()]] - code - tests/test_config.py
- [[test_trading_config_defaults_confidence_actionable_move_pct()]] - code - tests/test_config.py
- [[test_trading_config_defaults_cover_cash_rich_cap_and_after_hours_summary()]] - code - tests/test_config.py
- [[test_trading_config_reads_alpha_vantage_settings()]] - code - tests/test_config.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Configuration_Tests
SORT file.name ASC
```

## Connections to other communities
- 8 edges to [[_COMMUNITY_Broker Test Doubles]]
- 2 edges to [[_COMMUNITY_Market Data Loading]]

## Top bridge nodes
- [[DummyLogger_4]] - degree 12, connects to 2 communities
- [[config.py]] - degree 6, connects to 1 community
- [[test_bias_safe_historical_premarket_snapshot_does_not_use_full_day_volume()]] - degree 3, connects to 1 community
- [[test_fetch_forward_close_window_uses_next_three_trading_days()]] - degree 3, connects to 1 community
- [[test_point_in_time_universe_snapshot_does_not_truncate_snapshot_symbols()]] - degree 3, connects to 1 community