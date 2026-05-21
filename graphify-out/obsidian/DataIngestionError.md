---
source_file: "trading_system/data.py"
type: "code"
community: "Market Data Loading"
location: "L22"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Market_Data_Loading
---

# DataIngestionError

## Connections
- [[._build_symbol_universe_from_snapshot()]] - `calls` [EXTRACTED]
- [[._fetch_daily_bars()]] - `calls` [EXTRACTED]
- [[._fetch_daily_bars_alpha_vantage()]] - `calls` [EXTRACTED]
- [[.build_universe()]] - `calls` [EXTRACTED]
- [[.fetch_alpaca_daily_bars_bulk()]] - `calls` [EXTRACTED]
- [[.fetch_close_to_close_return()_1]] - `calls` [EXTRACTED]
- [[.fetch_forward_close_window()]] - `calls` [EXTRACTED]
- [[ConfidenceBucket]] - `uses` [INFERRED]
- [[ConfidenceCalibrator]] - `uses` [INFERRED]
- [[DummyLogger_4]] - `uses` [INFERRED]
- [[DummyLogger_6]] - `uses` [INFERRED]
- [[HistoricalDecisionOutcome]] - `uses` [INFERRED]
- [[IdentityCalibrator]] - `uses` [INFERRED]
- [[IndicatorSnapshot]] - `uses` [INFERRED]
- [[PremarketSnapshot]] - `uses` [INFERRED]
- [[RateLimiter]] - `uses` [INFERRED]
- [[RuntimeError]] - `inherits` [EXTRACTED]
- [[StubCalibrator]] - `uses` [INFERRED]
- [[SymbolMarketData]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[data.py]] - `contains` [EXTRACTED]
- [[fetch_top_us_companies_by_market_cap()]] - `calls` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Market_Data_Loading