from trading_system.llm import TokenUsage, TokenUsageTracker


def test_token_usage_tracker_aggregates_calls():
    tracker = TokenUsageTracker()
    tracker.record(
        stage="debate",
        usage=TokenUsage(
            provider="ollama",
            model="qwen3.5:4b",
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
        ),
        symbol="AAPL",
        role="bull",
    )
    tracker.record(
        stage="decision",
        usage=TokenUsage(
            provider="ollama",
            model="qwen3.5:9b",
            input_tokens=80,
            output_tokens=20,
            total_tokens=100,
        ),
    )

    payload = tracker.to_payload()

    assert payload["totals"] == {
        "call_count": 2,
        "input_tokens": 200,
        "output_tokens": 50,
        "total_tokens": 250,
    }
    assert payload["by_stage"]["debate"]["total_tokens"] == 150
    assert payload["by_model"]["ollama:qwen3.5:9b"]["call_count"] == 1


def test_token_usage_tracker_keeps_logprob_fields():
    tracker = TokenUsageTracker()
    tracker.record(
        stage="decision",
        usage=TokenUsage(
            provider="ollama",
            model="qwen3.5:9b",
            input_tokens=50,
            output_tokens=10,
            total_tokens=60,
            average_logprob=-0.4,
            average_probability=0.67,
        ),
    )

    payload = tracker.to_payload()

    assert payload["calls"][0]["average_logprob"] == -0.4
    assert payload["calls"][0]["average_probability"] == 0.67
