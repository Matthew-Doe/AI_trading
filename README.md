# AI Trading System

Production-structured local Python trading system for daily pre-open idea generation, AI debate, final decisioning, explicit LLM token tracking, and Alpaca paper-order execution. The default configuration scans 200 symbols and debates the top 20.

Index exposure is supported through tradable proxy ETFs such as `SPY`, `QQQ`, `DIA`, and `IWM`.

## Architecture

- `trading_system/data.py`: dynamic top-500 universe, OHLCV ingestion, premarket snapshot, indicators, optional news
- `trading_system/selection.py`: tunable scoring model for candidate selection
- `trading_system/debate.py`: sequential bull/bear debates through Ollama, OpenAI, or Anthropic with strict JSON validation
- `trading_system/decision.py`: final portfolio decision layer with normalized allocations
- `trading_system/llm.py`: provider abstraction, Ollama logprob capture, and normalized token usage tracking
- `trading_system/execution.py`: Alpaca paper-trading order planning, held-position evaluation, and submission safeguards
- `trading_system/main.py`: end-to-end orchestrator, replay mode, mock mode, artifact writing, CLI summary
- `trading_system/scheduler.py`: APScheduler-based daily runner

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in Alpaca paper credentials.
4. Configure one LLM provider.

Ollama:

```bash
ollama serve
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

OpenAI:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export OPENAI_DEBATE_MODEL=your_openai_debate_model
export OPENAI_DECISION_MODEL=your_openai_decision_model
```

Anthropic:

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key_here
export ANTHROPIC_DEBATE_MODEL=your_anthropic_debate_model
export ANTHROPIC_DECISION_MODEL=your_anthropic_decision_model
```

5. Leave `EXECUTE_ORDERS=false` until you are ready to let the paper account submit orders.

## Running

Mock run with no external APIs:

```bash
python -m trading_system.main --mock
```

Run with provider/API-key overrides for a single invocation:

```bash
python -m trading_system.main \
  --llm-provider openai \
  --openai-api-key your_key_here \
  --llm-debate-model your_openai_debate_model \
  --llm-decision-model your_openai_decision_model
```

Live paper run:

```bash
python -m trading_system.main
```

Optional news:

```bash
python -m trading_system.main --include-news
```

Replay a prior run:

```bash
python -m trading_system.main --replay-run-id 20260414T123000Z
```

Send only the market-close Telegram summary:

```bash
python -m trading_system.main --market-close-summary
```

## Scheduler

Run the built-in scheduler:

```bash
python -m trading_system.scheduler
```

Or schedule with cron for `8:30 AM`, `12:30 PM`, and `3:30 PM` ET on weekdays:

```cron
30 8,12,15 * * 1-5 cd /home/matthew/devel/AI_trading && /path/to/venv/bin/python -m trading_system.main >> logs/cron.log 2>&1
```

The pipeline checks the NYSE calendar and exits without trading on non-market days.

The built-in scheduler uses `SCHEDULED_TIMES_ET`, which defaults to `08:30,12:30,15:30`.

The market-close Telegram summary uses `BENCHMARK_SYMBOL` and runs at `MARKET_CLOSE_SUMMARY_HOUR_ET:MARKET_CLOSE_SUMMARY_MINUTE_ET`, which defaults to `16:05` for `SPY`.

## Data Sources

- Universe: `companiesmarketcap.com` U.S. market-cap rankings, fetched dynamically
- Market data: Alpaca market data when credentials are present, with `yfinance` fallback
- Optional news: `yfinance`
- Execution and account state: `alpaca-py` paper trading

## Safety Controls

- Paper trading only
- No hardcoded tickers
- Configurable index proxy symbols added to the universe via `INDEX_PROXY_SYMBOLS`
- Sequential debate execution, no threads
- Confidence thresholding before execution
- Max single new trade capped at 1% of account equity by default
- High-confidence long trades can raise the single-trade cap to 10% of cash only after Telegram approval
- Open pending orders are reviewed before the market opens and can be cancelled if extended-hours price movement exceeds a configured threshold
- Held-position signals: `buy_more`, `sell`, `hold`
- Existing position and open-order checks
- Max total exposure cap
- Max per-trade risk budget based on ATR/price stop proxy
- Shared request limiter capped at 100 external requests per minute
- Fail-safe run termination on critical failures

## Artifacts

Each run writes to `runs/<run_id>/`:

- `universe.json`
- `selected_symbols.json`
- `debates.json`
- `decisions.json`
- `order_plans.json`
- `held_position_signals.json`
- `execution_results.json`
- `llm_usage.json`
- `summary.txt`

Structured logs are written to `logs/<run_id>.log`.

Cached universe and symbol snapshots are written under `.cache/` to reduce repeated external calls and tolerate transient data-source failures.

## Index Proxies

The system cannot submit an order for a raw market index, but it can trade liquid ETFs that track those indexes. By default the universe is extended with:

- `SPY` for the S&P 500
- `QQQ` for the Nasdaq-100
- `DIA` for the Dow Jones Industrial Average
- `IWM` for the Russell 2000

Override the default set with:

```bash
export INDEX_PROXY_SYMBOLS=SPY,QQQ,VTI
```

## Notes

- `companiesmarketcap.com` parsing may require maintenance if the site changes its markup.
- Intraday/premarket data availability depends on Yahoo Finance coverage.
- Alpaca shorting support in paper mode can vary by symbol and account settings; the code gates shorts with `ALLOW_SHORTING`.
- The system defaults to dry-run order behavior even in live mode unless `EXECUTE_ORDERS=true`.
- Token usage is recorded for every LLM call and summarized in both `llm_usage.json` and `summary.txt`.
- On Ollama, `/api/generate` supports `logprobs`; the final decision layer uses average output probability as a confidence cap before execution decisions are made.
- Telegram receives an end-of-run token summary including input, output, total, and tokens-per-second rates.
- Telegram can also send a daily market-close summary comparing portfolio return to `SPY`.
- Pending open orders are reviewed before `9:30 AM ET` using extended-hours prices; orders with moves above `PENDING_ORDER_REVIEW_MAX_GAP_PCT` are cancelled.

## Telegram

To send order summaries and approve oversized high-confidence long orders from Telegram, set:

```bash
export TELEGRAM_BOT_TOKEN=your_bot_token
export TELEGRAM_CHAT_ID=your_chat_id
export TELEGRAM_APPROVAL_TIMEOUT_SECONDS=300
```

Behavior:

- After each submitted or dry-run order, the bot sends a summary message.
- Standard new `long` orders can scale from the default `1%` cap up to `5%` of available cash when cash is at least `20%` of account equity.
- For new `long` orders with confidence `>= 0.95`, the system can expand the cap from `1%` to `10%` of cash.
- The larger cap is used only if you tap the Telegram `Approve` button before timeout.
- Without approval, the order stays at the normal `MAX_SINGLE_TRADE_PCT` cap.
- At market close, the bot can send a daily summary comparing portfolio performance to `SPY`.
