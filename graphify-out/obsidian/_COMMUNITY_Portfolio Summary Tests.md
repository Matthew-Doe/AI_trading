---
type: community
cohesion: 0.11
members: 30
---

# Portfolio Summary Tests

**Cohesion:** 0.11 - loosely connected
**Members:** 30 nodes

## Members
- [[.__init__()]] - code - tests/test_portfolio_summary.py
- [[.__init__()_12]] - code - trading_system/portfolio_summary.py
- [[.build_summary()]] - code - trading_system/portfolio_summary.py
- [[.error()]] - code - tests/test_portfolio_summary.py
- [[.fetch_close_to_close_return()]] - code - tests/test_portfolio_summary.py
- [[.get_account()]] - code - tests/test_portfolio_summary.py
- [[.get_all_positions()]] - code - tests/test_portfolio_summary.py
- [[.info()]] - code - tests/test_portfolio_summary.py
- [[.is_enabled()]] - code - tests/test_portfolio_summary.py
- [[.send_message()]] - code - tests/test_portfolio_summary.py
- [[.send_summary()]] - code - trading_system/portfolio_summary.py
- [[.warning()]] - code - tests/test_portfolio_summary.py
- [[DummyLogger]] - code - tests/test_portfolio_summary.py
- [[FakeMarketData]] - code - tests/test_portfolio_summary.py
- [[FakeTelegram]] - code - tests/test_portfolio_summary.py
- [[FakeTradingClient]] - code - tests/test_portfolio_summary.py
- [[MarketCloseReporter]] - code - trading_system/portfolio_summary.py
- [[MarketCloseSummary]] - code - trading_system/portfolio_summary.py
- [[_parse_hhmm()]] - code - trading_system/scheduler.py
- [[build_market_close_summary_message()]] - code - trading_system/portfolio_summary.py
- [[live_paper_phase_schedule()]] - code - trading_system/scheduler.py
- [[main()_3]] - code - trading_system/scheduler.py
- [[portfolio_summary.py]] - code - trading_system/portfolio_summary.py
- [[scheduler.py]] - code - trading_system/scheduler.py
- [[send_market_close_summary()]] - code - trading_system/portfolio_summary.py
- [[test_build_market_close_summary_message_uses_relative_performance()]] - code - tests/test_portfolio_summary.py
- [[test_live_paper_phase_schedule_uses_entry_and_exit_times()]] - code - tests/test_scheduler.py
- [[test_market_close_reporter_sends_telegram_message()]] - code - tests/test_portfolio_summary.py
- [[test_portfolio_summary.py]] - code - tests/test_portfolio_summary.py
- [[test_scheduler.py]] - code - tests/test_scheduler.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Portfolio_Summary_Tests
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Broker Test Doubles]]
- 3 edges to [[_COMMUNITY_Market Data Loading]]
- 3 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 3 edges to [[_COMMUNITY_Live Execution Controls]]

## Top bridge nodes
- [[MarketCloseReporter]] - degree 13, connects to 3 communities
- [[MarketCloseSummary]] - degree 10, connects to 3 communities
- [[main()_3]] - degree 6, connects to 2 communities
- [[.__init__()_12]] - degree 3, connects to 2 communities
- [[DummyLogger]] - degree 8, connects to 1 community