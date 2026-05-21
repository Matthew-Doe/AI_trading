---
type: community
cohesion: 0.67
members: 4
---

# Daily Rollout Plan

**Cohesion:** 0.67 - moderately connected
**Members:** 4 nodes

## Members
- [[Live Paper Daily Rollout Design]] - document - docs/superpowers/specs/2026-05-01-live-paper-daily-rollout-design.md
- [[Live Paper Daily Rollout Implementation Plan]] - document - docs/superpowers/plans/2026-05-01-live-paper-daily-rollout.md
- [[Live Paper Daily Runner]] - code - scripts/run_live_paper_daily.py
- [[Live Paper Start Date Guard]] - code - scripts/run_live_paper_daily.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Daily_Rollout_Plan
SORT file.name ASC
```
