# January Profit Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-gated strategy experiments that can improve January profit factor, net PnL, drawdown, PnL concentration, and traded volume without changing the baseline path.

**Architecture:** Keep baseline behavior unchanged when all experiment flags are disabled. Add small, testable hooks in `BacktestExecutionEngine` for confidence sizing, conditional thesis-failure hold extension, partial target exits, and tax-adjusted re-entry. Use January A/B backtests to decide which flags remain useful.

**Tech Stack:** Python dataclasses, existing `TradingConfig`, pytest, existing `backtest_engine.py` report analytics.

---

### Task 1: Confidence Sizing Experiment

**Files:**
- Modify: `trading_system/config.py`
- Modify: `trading_system/backtest_execution.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`
- Test: `tests/test_backtest_execution.py`

- [ ] Add disabled-by-default `ENABLE_CONFIDENCE_SIZING_EXPERIMENT`.
- [ ] Add thresholds/multipliers for full-size, mid-size, and low-confidence skip behavior.
- [ ] Apply sizing multiplier to allocation, risk budget, and trade cap in `_size_position`.
- [ ] Lower high-confidence full-size threshold to the experiment threshold only when the experiment is enabled.
- [ ] Test default behavior is unchanged.
- [ ] Test `0.70-0.90` confidence gets half-sized.
- [ ] Test below-band confidence produces zero quantity.
- [ ] Run targeted tests and commit.

### Task 2: Conditional Thesis-Failure Hold Extension

**Files:**
- Modify: `trading_system/config.py`
- Modify: `trading_system/backtest_execution.py`
- Modify: `.env.example`
- Test: `tests/test_backtest_execution.py`

- [ ] Add hold-extension observation count and adverse-move limit config.
- [ ] Track thesis-failure deferrals on positions.
- [ ] When `ENABLE_CONDITIONAL_HOLD_EXTENSION=true`, defer thesis-failure exits for the configured observations unless the loss exceeds the adverse-move limit.
- [ ] Log deferrals in `exit_adjustment_logs`.
- [ ] Test deferral, eventual exit, and adverse-move immediate exit.
- [ ] Run targeted tests and commit.

### Task 3: Partial Target Exits

**Files:**
- Modify: `trading_system/config.py`
- Modify: `trading_system/backtest_execution.py`
- Modify: `.env.example`
- Test: `tests/test_backtest_execution.py`

- [ ] Add partial target fraction and trailing stop config.
- [ ] When `ENABLE_PARTIAL_PROFIT_TAKING=true`, sell only the configured fraction on first target hit.
- [ ] Keep the remaining position open with stop ratcheted to breakeven or a trailing-profit stop.
- [ ] Record the partial exit as a trade with `exit_reason=partial_target_hit`.
- [ ] Let later target/stop/hold exits close the residual quantity normally.
- [ ] Test cash, residual quantity, stop ratchet, and trade records.
- [ ] Run targeted tests and commit.

### Task 4: Tax-Adjusted Re-Entry

**Files:**
- Modify: `trading_system/config.py`
- Modify: `trading_system/backtest_execution.py`
- Modify: `.env.example`
- Test: `tests/test_backtest_execution.py`

- [ ] Add tax re-entry minimum confidence, EV, and size multiplier config.
- [ ] When `ENABLE_TAX_ADJUSTED_EV_REENTRY=true`, allow a tax-blocked decision only if confidence and expected value clear configured thresholds.
- [ ] Downsize allowed tax re-entries to limit wash-sale risk and record the sizing reason.
- [ ] Continue recording tax shadows for all blocked decisions.
- [ ] Test blocked default, allowed re-entry, and downsized re-entry.
- [ ] Run targeted tests and commit.

### Task 5: A/B Evaluation

**Files:**
- Use: `backtest_engine.py`
- Use: `backtests/*/backtest_report.json`

- [ ] Run baseline confirmation only if current baseline report is missing.
- [ ] Run January A/B for each experiment independently.
- [ ] Run combined January A/B for experiments that independently improve metrics.
- [ ] Compare net PnL, profit factor, max drawdown, top-winner concentration, and traded notional.
- [ ] Keep only changes that improve the metric set or leave them disabled with documented results.
