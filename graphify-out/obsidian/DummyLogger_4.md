---
source_file: "tests/test_config.py"
type: "code"
community: "Configuration Tests"
location: "L12"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Configuration_Tests
---

# DummyLogger

## Connections
- [[.error()_1]] - `method` [EXTRACTED]
- [[.info()_3]] - `method` [EXTRACTED]
- [[.warning()_4]] - `method` [EXTRACTED]
- [[DataIngestionError]] - `uses` [INFERRED]
- [[MarketDataService]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[test_bias_safe_historical_premarket_snapshot_does_not_use_full_day_volume()]] - `calls` [EXTRACTED]
- [[test_config.py]] - `contains` [EXTRACTED]
- [[test_fetch_forward_close_window_uses_next_three_trading_days()]] - `calls` [EXTRACTED]
- [[test_point_in_time_universe_snapshot_does_not_truncate_snapshot_symbols()]] - `calls` [EXTRACTED]
- [[test_point_in_time_universe_snapshot_fails_closed_without_snapshot()]] - `calls` [EXTRACTED]
- [[test_point_in_time_universe_snapshot_selects_latest_on_or_before_date()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Configuration_Tests