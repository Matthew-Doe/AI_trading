---
source_file: "trading_system/portfolio_summary.py"
type: "code"
community: "Portfolio Summary Tests"
location: "L29"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Portfolio_Summary_Tests
---

# MarketCloseReporter

## Connections
- [[.__init__()_12]] - `method` [EXTRACTED]
- [[.build_summary()]] - `method` [EXTRACTED]
- [[.send_summary()]] - `method` [EXTRACTED]
- [[DummyLogger]] - `uses` [INFERRED]
- [[FakeMarketData]] - `uses` [INFERRED]
- [[FakeTelegram]] - `uses` [INFERRED]
- [[FakeTradingClient]] - `uses` [INFERRED]
- [[MarketDataService]] - `uses` [INFERRED]
- [[TelegramNotifier]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[portfolio_summary.py]] - `contains` [EXTRACTED]
- [[send_market_close_summary()]] - `calls` [EXTRACTED]
- [[test_market_close_reporter_sends_telegram_message()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Portfolio_Summary_Tests