---
type: community
cohesion: 0.08
members: 49
---

# LLM Debate Handling

**Cohesion:** 0.08 - loosely connected
**Members:** 49 nodes

## Members
- [[.__init__()_10]] - code - trading_system/data.py
- [[.__init__()_18]] - code - trading_system/debate.py
- [[.__init__()_8]] - code - trading_system/llm.py
- [[.__init__()_7]] - code - trading_system/llm.py
- [[.__init__()_16]] - code - trading_system/utils.py
- [[._build_role_prompt()]] - code - trading_system/debate.py
- [[._generate_anthropic()]] - code - trading_system/llm.py
- [[._generate_ollama()]] - code - trading_system/llm.py
- [[._generate_openai()]] - code - trading_system/llm.py
- [[._generate_valid_debate_json()]] - code - trading_system/debate.py
- [[._group_by()]] - code - trading_system/llm.py
- [[._run_role()]] - code - trading_system/debate.py
- [[.acquire()]] - code - trading_system/utils.py
- [[.generate()]] - code - trading_system/llm.py
- [[.record()]] - code - trading_system/llm.py
- [[.run_debate_for_symbol()]] - code - trading_system/debate.py
- [[.supports_warmup()]] - code - trading_system/llm.py
- [[.to_payload()]] - code - trading_system/llm.py
- [[.warmup()]] - code - trading_system/debate.py
- [[.warning()_1]] - code - tests/test_debate.py
- [[DebateError]] - code - trading_system/debate.py
- [[DebateResult]] - code - trading_system/models.py
- [[DummyLogger_1]] - code - tests/test_debate.py
- [[LLMClient]] - code - trading_system/llm.py
- [[LLMError]] - code - trading_system/llm.py
- [[LLMResponse]] - code - trading_system/llm.py
- [[OllamaDebateEngine]] - code - trading_system/debate.py
- [[RateLimiter]] - code - trading_system/utils.py
- [[RunArtifacts]] - code - trading_system/models.py
- [[SymbolDebate]] - code - trading_system/models.py
- [[TokenUsage]] - code - trading_system/llm.py
- [[TokenUsageRecord]] - code - trading_system/llm.py
- [[TokenUsageTracker]] - code - trading_system/llm.py
- [[_aggregate()]] - code - trading_system/llm.py
- [[_build_json_repair_prompt()_1]] - code - trading_system/debate.py
- [[_coerce_message_text()]] - code - trading_system/llm.py
- [[_extract_json()_1]] - code - trading_system/debate.py
- [[_llm_generate()]] - code - trading_system/debate.py
- [[_summarize_logprobs()]] - code - trading_system/llm.py
- [[_validate_debate_payload()]] - code - trading_system/debate.py
- [[debate.py]] - code - trading_system/debate.py
- [[llm.py]] - code - trading_system/llm.py
- [[models.py]] - code - trading_system/models.py
- [[test_debate.py]] - code - tests/test_debate.py
- [[test_extract_json_finds_embedded_object()]] - code - tests/test_debate.py
- [[test_extract_json_raises_when_no_object_present()]] - code - tests/test_debate.py
- [[test_llm.py]] - code - tests/test_llm.py
- [[test_token_usage_tracker_aggregates_calls()]] - code - tests/test_llm.py
- [[test_token_usage_tracker_keeps_logprob_fields()]] - code - tests/test_llm.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/LLM_Debate_Handling
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Run Reporting Pipeline]]
- 8 edges to [[_COMMUNITY_Market Data Loading]]
- 8 edges to [[_COMMUNITY_Broker Test Doubles]]
- 6 edges to [[_COMMUNITY_Decision Engine Tests]]
- 5 edges to [[_COMMUNITY_Live Execution Controls]]
- 1 edge to [[_COMMUNITY_Backtest Execution Tests]]
- 1 edge to [[_COMMUNITY_Confidence Calibration Tests]]
- 1 edge to [[_COMMUNITY_Candidate Selection]]

## Top bridge nodes
- [[OllamaDebateEngine]] - degree 16, connects to 3 communities
- [[LLMClient]] - degree 15, connects to 3 communities
- [[TokenUsageTracker]] - degree 14, connects to 3 communities
- [[DebateError]] - degree 12, connects to 3 communities
- [[models.py]] - degree 10, connects to 3 communities