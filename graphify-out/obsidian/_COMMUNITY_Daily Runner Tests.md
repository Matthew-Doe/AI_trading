---
type: community
cohesion: 0.29
members: 7
---

# Daily Runner Tests

**Cohesion:** 0.29 - loosely connected
**Members:** 7 nodes

## Members
- [[test_invalid_start_date_exits()]] - code - tests/test_run_live_paper_daily.py
- [[test_main_forwards_full_phase_by_default()]] - code - tests/test_run_live_paper_daily.py
- [[test_main_preserves_explicit_phase_and_extra_args()]] - code - tests/test_run_live_paper_daily.py
- [[test_main_skips_before_start_date()]] - code - tests/test_run_live_paper_daily.py
- [[test_run_live_paper_daily.py]] - code - tests/test_run_live_paper_daily.py
- [[test_should_run_is_false_before_start_date()]] - code - tests/test_run_live_paper_daily.py
- [[test_should_run_is_true_on_start_date()]] - code - tests/test_run_live_paper_daily.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Daily_Runner_Tests
SORT file.name ASC
```
