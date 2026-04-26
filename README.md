# AI Trading System

Local Python system for AI-assisted paper-trading research. It builds a U.S. equity universe, ranks candidates, runs bull/bear LLM debates, generates final trade decisions, simulates or submits Alpaca paper orders, and writes auditable run artifacts.

This is research software, not financial advice. Backtest results should be treated as exploratory until the data, execution, and calibration assumptions are independently reviewed.

## Main Modules

- `trading_system/data.py`: universe, OHLCV, indicators, premarket data, data-quality checks
- `trading_system/selection.py`: candidate scoring and ranking
- `trading_system/debate.py`: bull/bear LLM debate generation
- `trading_system/decision.py`: final decision JSON validation, calibration, allocation normalization
- `trading_system/execution.py`: Alpaca paper order planning and safeguards
- `trading_system/backtest_execution.py`: portfolio-style backtest execution simulation
- `backtest_engine.py`: chronological backtest runner
- `trading_system/dashboard.py`: local read-only dashboard

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with Alpaca paper credentials and one LLM provider. The default provider is Ollama.

```bash
ollama serve
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

Keep `EXECUTE_ORDERS=false` unless you intentionally want paper orders submitted.

## Common Commands

Mock run:

```bash
python -m trading_system.main --mock
```

Daily pipeline:

```bash
python -m trading_system.main
```

Backtest:

```bash
python backtest_engine.py --start 2026-03-01 --end 2026-04-20
```

Dashboard:

```bash
python -m trading_system.dashboard
```

Open `http://127.0.0.1:8765`.

Tests:

```bash
python -m pytest -q
```

## Configuration

Most behavior is controlled by `.env`. Important settings include:

- `CANDIDATE_COUNT`: number of selected symbols debated each day
- `MIN_CONFIDENCE`: minimum calibrated confidence before action
- `MIN_REWARD_RISK_RATIO`: minimum reward/risk for actionable trades
- `MAX_SINGLE_TRADE_PCT`: default new-position cap
- `CASH_RICH_TRADE_PCT`: larger cap when cash is abundant
- `MAX_TOTAL_EXPOSURE`: portfolio exposure cap
- `MAX_DAILY_LOSS_PCT`: daily loss kill switch
- `INDEX_PROXY_SYMBOLS`: tradable index proxies, default `SPY,QQQ,DIA,IWM`
- `EXECUTE_ORDERS`: must be `true` to submit paper orders

See `.env.example` for the full list.

## Artifacts

Pipeline runs write to `runs/<run_id>/`. Backtests write to `backtests/<run_id>/`.

Important files:

- `selected_symbols.json`
- `debates.json`
- `decisions.json`
- `order_plans.json`
- `execution_results.json`
- `backtest_report.json`
- `report.json` / `report.html`
- `summary.txt`

Logs are written under `logs/`. Cached market data is written under `.cache/`.

## Safety Controls

- Dry-run by default
- Paper-trading Alpaca endpoint by default
- Market-data quality flags
- JSON schema validation for LLM outputs
- Confidence thresholding and calibration
- Max trade, exposure, weight, risk, and daily-loss caps
- Existing-position and open-order checks
- Optional Telegram approval for oversized high-confidence longs

## Backtests

The repository includes one main completed backtest under:

```text
backtests/20260426T040610Z/
```

That run covers `2026-03-01` through `2026-04-20` and finished at approximately `+15.31%` equity return. Treat it as a test artifact, not proof of strategy robustness. Known caveats include current-universe bias risk, daily-close stop/target simulation, calibration assumptions, and LLM output variability.

## External Review

`external_review/request.md` contains a short prompt for outside reviewers. The main questions are whether confidence accuracy, profitability, logging, backtest realism, and auditability can be improved.
