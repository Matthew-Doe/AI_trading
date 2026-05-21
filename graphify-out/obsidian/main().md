---
source_file: "preload_market_data.py"
type: "code"
community: "Market Data Loading"
location: "L88"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Market_Data_Loading
---

# main()

## Connections
- [[MarketDataService]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[_parse_utc_date()]] - `calls` [EXTRACTED]
- [[get_logger()]] - `calls` [INFERRED]
- [[load_symbols_from_snapshot()]] - `calls` [EXTRACTED]
- [[parse_args()]] - `calls` [EXTRACTED]
- [[preload_alpaca_daily_bars()]] - `calls` [EXTRACTED]
- [[preload_market_data.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Market_Data_Loading