---
source_file: "trading_system/data.py"
type: "code"
community: "Market Data Loading"
location: "L26"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Market_Data_Loading
---

# MarketDataService

## Connections
- [[.__init__()_10]] - `method` [EXTRACTED]
- [[.__init__()_9]] - `calls` [INFERRED]
- [[.__init__()_12]] - `calls` [INFERRED]
- [[._build_price_summary()]] - `method` [EXTRACTED]
- [[._build_symbol_universe()]] - `method` [EXTRACTED]
- [[._build_symbol_universe_from_cache()]] - `method` [EXTRACTED]
- [[._build_symbol_universe_from_snapshot()]] - `method` [EXTRACTED]
- [[._compute_indicators()]] - `method` [EXTRACTED]
- [[._data_quality_flags()]] - `method` [EXTRACTED]
- [[._fetch_daily_bars()]] - `method` [EXTRACTED]
- [[._fetch_daily_bars_alpaca()]] - `method` [EXTRACTED]
- [[._fetch_daily_bars_alpha_vantage()]] - `method` [EXTRACTED]
- [[._fetch_news_headlines()]] - `method` [EXTRACTED]
- [[._fetch_premarket_snapshot()]] - `method` [EXTRACTED]
- [[._fetch_premarket_snapshot_alpaca()]] - `method` [EXTRACTED]
- [[._read_daily_bars_cache()]] - `method` [EXTRACTED]
- [[._read_json_cache()]] - `method` [EXTRACTED]
- [[._read_symbol_cache()]] - `method` [EXTRACTED]
- [[._write_symbol_cache()]] - `method` [EXTRACTED]
- [[._yf_download()]] - `method` [EXTRACTED]
- [[.apply_data_quality()]] - `method` [EXTRACTED]
- [[.build_universe()]] - `method` [EXTRACTED]
- [[.fetch_alpaca_daily_bars_bulk()]] - `method` [EXTRACTED]
- [[.fetch_close_to_close_return()_1]] - `method` [EXTRACTED]
- [[.fetch_forward_close_window()]] - `method` [EXTRACTED]
- [[.is_market_day()]] - `method` [EXTRACTED]
- [[.write_daily_bars_cache()]] - `method` [EXTRACTED]
- [[DecisionEngine]] - `uses` [INFERRED]
- [[DecisionError]] - `uses` [INFERRED]
- [[DummyLogger_2]] - `uses` [INFERRED]
- [[DummyLogger_3]] - `uses` [INFERRED]
- [[DummyLogger_4]] - `uses` [INFERRED]
- [[IndicatorSnapshot]] - `uses` [INFERRED]
- [[MarketCloseReporter]] - `uses` [INFERRED]
- [[MarketCloseSummary]] - `uses` [INFERRED]
- [[PremarketSnapshot]] - `uses` [INFERRED]
- [[RateLimiter]] - `uses` [INFERRED]
- [[SymbolMarketData]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[data.py]] - `contains` [EXTRACTED]
- [[main()]] - `calls` [INFERRED]
- [[run_backtest()]] - `calls` [INFERRED]
- [[run_pipeline()]] - `calls` [INFERRED]
- [[test_alpha_vantage_daily_bars_parse_ohlcv_response()]] - `calls` [INFERRED]
- [[test_cached_market_data_is_rechecked_for_quality()]] - `calls` [INFERRED]
- [[test_confidence_calibrator_returns_raw_when_calibration_disabled()]] - `calls` [INFERRED]
- [[test_data_quality_allows_clean_symbol()]] - `calls` [INFERRED]
- [[test_data_quality_marks_extreme_discontinuity_untradeable()]] - `calls` [INFERRED]
- [[test_fetch_alpaca_daily_bars_bulk_paginates_and_groups_symbols()]] - `calls` [INFERRED]
- [[test_fetch_daily_bars_prefers_local_daily_bar_cache()]] - `calls` [INFERRED]
- [[test_preload_alpaca_daily_bars_writes_symbol_cache_files()]] - `calls` [INFERRED]
- [[test_symbol_universe_appends_index_proxies_without_duplicates()]] - `calls` [INFERRED]
- [[test_symbol_universe_falls_back_to_cached_symbol_files()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Market_Data_Loading