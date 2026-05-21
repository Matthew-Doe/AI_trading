---
type: community
cohesion: 0.22
members: 19
---

# Weekly Review Flow

**Cohesion:** 0.22 - loosely connected
**Members:** 19 nodes

## Members
- [[_adaptation_candidates()]] - code - trading_system/weekly_review.py
- [[_breakdown()]] - code - trading_system/weekly_review.py
- [[_confidence_bucket()]] - code - trading_system/weekly_review.py
- [[_merge_dicts()]] - code - trading_system/weekly_review.py
- [[_performance()]] - code - trading_system/weekly_review.py
- [[_regime_breakdown()]] - code - trading_system/weekly_review.py
- [[_report()]] - code - tests/test_weekly_review.py
- [[_symbol_concentration()]] - code - trading_system/weekly_review.py
- [[build_weekly_review()]] - code - trading_system/weekly_review.py
- [[load_reports()]] - code - trading_system/weekly_review.py
- [[main()_2]] - code - scripts/weekly_trade_review.py
- [[render_weekly_markdown()]] - code - trading_system/weekly_review.py
- [[test_weekly_review.py]] - code - tests/test_weekly_review.py
- [[test_weekly_review_flags_adaptation_candidates_and_renders_markdown()]] - code - tests/test_weekly_review.py
- [[test_weekly_review_summarizes_performance_breakdowns_and_contributions()]] - code - tests/test_weekly_review.py
- [[test_write_weekly_review_outputs_json_and_markdown()]] - code - tests/test_weekly_review.py
- [[weekly_review.py]] - code - trading_system/weekly_review.py
- [[weekly_trade_review.py]] - code - scripts/weekly_trade_review.py
- [[write_weekly_review()]] - code - trading_system/weekly_review.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Weekly_Review_Flow
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 1 edge to [[_COMMUNITY_Market Data Loading]]

## Top bridge nodes
- [[write_weekly_review()]] - degree 7, connects to 1 community
- [[load_reports()]] - degree 3, connects to 1 community