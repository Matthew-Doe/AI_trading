---
source_file: "tests/test_portfolio_summary.py"
type: "code"
community: "Portfolio Summary Tests"
location: "L69"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Portfolio_Summary_Tests
---

# test_market_close_reporter_sends_telegram_message()

## Connections
- [[DummyLogger]] - `calls` [EXTRACTED]
- [[FakeMarketData]] - `calls` [EXTRACTED]
- [[FakeTelegram]] - `calls` [EXTRACTED]
- [[FakeTradingClient]] - `calls` [EXTRACTED]
- [[MarketCloseReporter]] - `calls` [INFERRED]
- [[TradingConfig]] - `calls` [INFERRED]
- [[test_portfolio_summary.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Portfolio_Summary_Tests