---
type: community
cohesion: 1.00
members: 2
---

# Preload Cache Contract

**Cohesion:** 1.00 - tightly connected
**Members:** 2 nodes

## Members
- [[Daily Bar Cache Contract]] - code - tests/test_data.py
- [[Preload Market Data Cache Contract]] - code - tests/test_preload_market_data.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Preload_Cache_Contract
SORT file.name ASC
```
