---
type: community
cohesion: 0.26
members: 15
---

# Candidate Selection

**Cohesion:** 0.26 - loosely connected
**Members:** 15 nodes

## Members
- [[.__init__()_5]] - code - trading_system/selection.py
- [[._breakout_score()]] - code - trading_system/selection.py
- [[._overextension_penalty()]] - code - trading_system/selection.py
- [[._trend_quality()]] - code - trading_system/selection.py
- [[.info()_6]] - code - tests/test_selection.py
- [[.score_symbol()]] - code - trading_system/selection.py
- [[.select()]] - code - trading_system/selection.py
- [[CandidateSelector]] - code - trading_system/selection.py
- [[DummyLogger_7]] - code - tests/test_selection.py
- [[clamp()]] - code - trading_system/utils.py
- [[selection.py]] - code - trading_system/selection.py
- [[test_candidate_selection_excludes_untradeable_symbols()]] - code - tests/test_selection.py
- [[test_candidate_selection_penalizes_overextended_symbols()]] - code - tests/test_selection.py
- [[test_candidate_selection_returns_ranked_subset()]] - code - tests/test_selection.py
- [[test_selection.py]] - code - tests/test_selection.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Candidate_Selection
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Decision Engine Tests]]
- 3 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 3 edges to [[_COMMUNITY_Broker Test Doubles]]
- 2 edges to [[_COMMUNITY_Market Data Loading]]
- 1 edge to [[_COMMUNITY_LLM Debate Handling]]

## Top bridge nodes
- [[clamp()]] - degree 10, connects to 3 communities
- [[CandidateSelector]] - degree 14, connects to 2 communities
- [[test_candidate_selection_excludes_untradeable_symbols()]] - degree 4, connects to 1 community
- [[test_candidate_selection_penalizes_overextended_symbols()]] - degree 4, connects to 1 community
- [[test_candidate_selection_returns_ranked_subset()]] - degree 4, connects to 1 community