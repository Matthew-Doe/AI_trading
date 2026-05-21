---
type: community
cohesion: 0.70
members: 5
---

# Daily Runner CLI

**Cohesion:** 0.70 - tightly connected
**Members:** 5 nodes

## Members
- [[_main_args()]] - code - scripts/run_live_paper_daily.py
- [[_start_date()]] - code - scripts/run_live_paper_daily.py
- [[main()_1]] - code - scripts/run_live_paper_daily.py
- [[run_live_paper_daily.py]] - code - scripts/run_live_paper_daily.py
- [[should_run()]] - code - scripts/run_live_paper_daily.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Daily_Runner_CLI
SORT file.name ASC
```
