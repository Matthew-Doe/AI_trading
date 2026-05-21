---
source_file: "trading_system/llm.py"
type: "code"
community: "LLM Debate Handling"
location: "L18"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/LLM_Debate_Handling
---

# TokenUsage

## Connections
- [[._generate_anthropic()]] - `calls` [EXTRACTED]
- [[._generate_ollama()]] - `calls` [EXTRACTED]
- [[._generate_openai()]] - `calls` [EXTRACTED]
- [[RateLimiter]] - `uses` [INFERRED]
- [[TradingConfig]] - `uses` [INFERRED]
- [[llm.py]] - `contains` [EXTRACTED]
- [[test_token_usage_tracker_aggregates_calls()]] - `calls` [INFERRED]
- [[test_token_usage_tracker_keeps_logprob_fields()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/LLM_Debate_Handling