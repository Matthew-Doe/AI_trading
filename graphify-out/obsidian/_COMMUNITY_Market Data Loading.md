---
type: community
cohesion: 0.08
members: 65
---

# Market Data Loading

**Cohesion:** 0.08 - loosely connected
**Members:** 65 nodes

## Members
- [[._build_price_summary()]] - code - trading_system/data.py
- [[._build_symbol_universe()]] - code - trading_system/data.py
- [[._build_symbol_universe_from_cache()]] - code - trading_system/data.py
- [[._build_symbol_universe_from_snapshot()]] - code - trading_system/data.py
- [[._compute_indicators()]] - code - trading_system/data.py
- [[._data_quality_flags()]] - code - trading_system/data.py
- [[._fetch_daily_bars()]] - code - trading_system/data.py
- [[._fetch_daily_bars_alpaca()]] - code - trading_system/data.py
- [[._fetch_daily_bars_alpha_vantage()]] - code - trading_system/data.py
- [[._fetch_news_headlines()]] - code - trading_system/data.py
- [[._fetch_premarket_snapshot()]] - code - trading_system/data.py
- [[._fetch_premarket_snapshot_alpaca()]] - code - trading_system/data.py
- [[._read_daily_bars_cache()]] - code - trading_system/data.py
- [[._read_json_cache()]] - code - trading_system/data.py
- [[._read_symbol_cache()]] - code - trading_system/data.py
- [[._write_symbol_cache()]] - code - trading_system/data.py
- [[._yf_download()]] - code - trading_system/data.py
- [[.apply_data_quality()]] - code - trading_system/data.py
- [[.build_universe()]] - code - trading_system/data.py
- [[.fetch_alpaca_daily_bars_bulk()]] - code - trading_system/data.py
- [[.fetch_close_to_close_return()_1]] - code - trading_system/data.py
- [[.fetch_forward_close_window()]] - code - trading_system/data.py
- [[.info()_2]] - code - tests/test_data.py
- [[.info()_1]] - code - tests/test_preload_market_data.py
- [[.is_market_day()]] - code - trading_system/data.py
- [[.to_prompt_payload()]] - code - trading_system/models.py
- [[.warning()_3]] - code - tests/test_data.py
- [[.warning()_2]] - code - tests/test_preload_market_data.py
- [[.write_daily_bars_cache()]] - code - trading_system/data.py
- [[DataIngestionError]] - code - trading_system/data.py
- [[DummyLogger_3]] - code - tests/test_data.py
- [[DummyLogger_2]] - code - tests/test_preload_market_data.py
- [[IndicatorSnapshot]] - code - trading_system/models.py
- [[MarketDataService]] - code - trading_system/data.py
- [[PremarketSnapshot]] - code - trading_system/models.py
- [[SymbolMarketData]] - code - trading_system/models.py
- [[_daily_bar_records_to_frame()]] - code - trading_system/data.py
- [[_parse_market_cap()]] - code - trading_system/data.py
- [[_parse_utc_date()]] - code - preload_market_data.py
- [[_symbol_market_data_from_cache()]] - code - trading_system/data.py
- [[_symbol_with_metrics()]] - code - tests/test_data.py
- [[_to_alpaca_symbol()]] - code - trading_system/data.py
- [[data.py]] - code - trading_system/data.py
- [[fetch_symbol_market_data()]] - code - trading_system/data.py
- [[fetch_top_us_companies_by_market_cap()]] - code - trading_system/data.py
- [[load_symbols_from_snapshot()]] - code - preload_market_data.py
- [[main()]] - code - preload_market_data.py
- [[parse_args()]] - code - preload_market_data.py
- [[preload_alpaca_daily_bars()]] - code - preload_market_data.py
- [[preload_market_data.py]] - code - preload_market_data.py
- [[read_json()]] - code - trading_system/utils.py
- [[safe_float()]] - code - trading_system/utils.py
- [[test_alpha_vantage_daily_bars_parse_ohlcv_response()]] - code - tests/test_data.py
- [[test_cached_market_data_is_rechecked_for_quality()]] - code - tests/test_data.py
- [[test_data.py]] - code - tests/test_data.py
- [[test_data_quality_allows_clean_symbol()]] - code - tests/test_data.py
- [[test_data_quality_marks_extreme_discontinuity_untradeable()]] - code - tests/test_data.py
- [[test_fetch_alpaca_daily_bars_bulk_paginates_and_groups_symbols()]] - code - tests/test_data.py
- [[test_fetch_daily_bars_prefers_local_daily_bar_cache()]] - code - tests/test_data.py
- [[test_load_mock_universe_includes_index_proxies()]] - code - tests/test_data.py
- [[test_preload_alpaca_daily_bars_writes_symbol_cache_files()]] - code - tests/test_preload_market_data.py
- [[test_preload_market_data.py]] - code - tests/test_preload_market_data.py
- [[test_symbol_universe_appends_index_proxies_without_duplicates()]] - code - tests/test_data.py
- [[test_symbol_universe_falls_back_to_cached_symbol_files()]] - code - tests/test_data.py
- [[utc_timestamp()]] - code - trading_system/utils.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Market_Data_Loading
SORT file.name ASC
```

## Connections to other communities
- 21 edges to [[_COMMUNITY_Broker Test Doubles]]
- 18 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 12 edges to [[_COMMUNITY_Backtest Execution Tests]]
- 8 edges to [[_COMMUNITY_LLM Debate Handling]]
- 6 edges to [[_COMMUNITY_Confidence Calibration Tests]]
- 5 edges to [[_COMMUNITY_Decision Engine Tests]]
- 3 edges to [[_COMMUNITY_Live Execution Controls]]
- 3 edges to [[_COMMUNITY_Portfolio Summary Tests]]
- 2 edges to [[_COMMUNITY_Configuration Tests]]
- 2 edges to [[_COMMUNITY_Candidate Selection]]
- 2 edges to [[_COMMUNITY_Dashboard API Tests]]
- 1 edge to [[_COMMUNITY_Backtest Report Builders]]
- 1 edge to [[_COMMUNITY_Weekly Review Flow]]

## Top bridge nodes
- [[MarketDataService]] - degree 53, connects to 7 communities
- [[SymbolMarketData]] - degree 24, connects to 7 communities
- [[DataIngestionError]] - degree 22, connects to 6 communities
- [[read_json()]] - degree 18, connects to 6 communities
- [[PremarketSnapshot]] - degree 12, connects to 4 communities