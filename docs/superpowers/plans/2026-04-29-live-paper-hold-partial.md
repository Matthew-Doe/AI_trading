# Live Paper Hold+Partial Implementation Plan

**Goal:** Run the January-tested `hold+partial` strategy on a brand-new Alpaca paper account using live paper orders that mirror the backtest behavior as closely as practical.

**Strategy Choice:** Use `hold+partial` for the first live paper rollout. Do not enable confidence sizing, tax-adjusted re-entry, or the combined `hold+tax+partial` strategy until there is enough paper-live evidence to justify another controlled comparison.

**Core Principle:** Make the live paper system behave like the backtest. Entries happen around the simulated open-entry window, exits are reviewed on a scheduled close-style pass, stops and targets are managed by the strategy rather than by intraday broker brackets, and larger trades do not require Telegram approval.

---

## Operating Assumptions

- The Alpaca account is a brand-new paper account.
- The system should fail closed if the account already has open positions or open orders unless explicitly configured otherwise.
- The strategy is paper-only. It should refuse to run against live Alpaca endpoints.
- `ENABLE_CONDITIONAL_HOLD_EXTENSION=true`.
- `ENABLE_PARTIAL_PROFIT_TAKING=true`.
- `ENABLE_TAX_ADJUSTED_EV_REENTRY=false`.
- Larger-trade approval should be bypassed in this paper backtest-style mode. The live sizing path should follow the same caps used by the backtest, not ask for permission.
- Use managed daily review exits, not Alpaca bracket orders, when matching backtest behavior is the priority.

---

## Files To Change

- `trading_system/config.py`: add live paper strategy flags, time-of-day settings, and paper-account safety guards.
- `.env.example`: document the exact environment variables for this mode.
- `trading_system/live_strategy_state.py`: add persistent JSON state for live positions, partial exits, and thesis-failure deferrals.
- `trading_system/execution.py`: adapt Alpaca order planning and held-position management to support backtest-style `hold+partial`.
- `trading_system/main.py`: route live runs through entry or exit-review phases.
- `trading_system/scheduler.py`: support separate entry and exit-review schedules.
- Tests under `tests/`: cover config parsing, persistent state, execution behavior, phase routing, and paper safety checks.

---

## Phase 1: Configuration And Safety

Add a disabled-by-default live paper mode with explicit strategy selection:

- `ENABLE_LIVE_PAPER_BACKTEST_STYLE`
- `LIVE_PAPER_STRATEGY=hold_partial`
- `REQUIRE_EMPTY_PAPER_ACCOUNT`
- `ALLOW_LIVE_LARGE_TRADE_APPROVAL=false`
- `LIVE_ENTRY_REVIEW_TIME_ET`
- `LIVE_EXIT_REVIEW_TIME_ET`

Validate time settings as `HH:MM`. Keep defaults conservative: entry review shortly after the market opens and exit review near the close.

Add paper-account safeguards:

- refuse non-paper Alpaca URLs or credentials in this mode
- fail if the account has positions or open orders and `REQUIRE_EMPTY_PAPER_ACCOUNT=true`
- clearly log the strategy mode, account mode, schedule phase, and approval behavior at startup

Checkpoint:

- config tests pass
- paper safety tests pass
- commit the config and safety changes

---

## Phase 2: Persistent Live Strategy State

Add a small JSON-backed state file for live paper strategy state. It should survive process restarts and record one entry per live position.

Track:

- symbol
- side
- quantity
- entry price
- stop price
- take-profit price
- entry timestamp
- whether partial profit has already been taken
- partial-exit quantity
- thesis-failure deferral count
- latest update timestamp

State responsibilities:

- record new entries after successful order placement or confirmed fill
- mark partial exits
- increment or reset thesis-failure deferrals
- clear closed positions
- reconcile state against actual Alpaca positions

Checkpoint:

- state persistence tests pass
- reconciliation tests pass
- commit the state module

---

## Phase 3: Backtest-Style Live Entry Planning

Update the Alpaca execution path so paper live entries use the same sizing concepts as the backtest:

- confidence
- allocation
- max single-trade cap
- risk-per-trade cap
- stop distance
- cash constraint
- max position weight

In paper backtest-style mode, do not use the Telegram larger-trade approval path. If the trade fits configured caps, it can be planned. If it does not, it should be reduced or skipped according to the same sizing constraints as the backtest.

Order style should be configurable, but the recommended starting setting is managed limit or market entry without broker bracket exits. The exit-review phase should own stop, target, hold-extension, and partial-exit behavior so the live path remains comparable to backtest results.

Checkpoint:

- order planning tests prove no Telegram approval call is made
- sizing tests compare live paper planning against representative backtest sizing cases
- commit entry-planning changes

---

## Phase 4: Conditional Thesis-Failure Hold

Port the backtest conditional hold behavior into live held-position review.

When a held position would normally exit for `thesis_failed`, allow a limited extension if:

- conditional hold is enabled
- unrealized loss is smaller than the configured adverse threshold
- the position has remaining deferrals

Persist deferral count in `LiveStrategyState`. The next review should see the updated count, even after restart.

If the adverse threshold is breached or the deferral limit is exhausted, plan the exit.

Checkpoint:

- tests cover first deferral, final allowed deferral, exhausted deferral, and adverse-threshold exit
- commit conditional hold changes

---

## Phase 5: Partial Target Exits

Port backtest partial target behavior into live held-position review.

When a position reaches its take-profit level:

- sell or cover the configured fraction
- keep the remaining position open
- move the stop according to the partial-profit trailing rule
- persist that the partial exit has happened
- avoid taking multiple partial exits for the same position

The live order plan should clearly mark partial exits so reporting can separate them from full closes.

Checkpoint:

- tests cover long and short partial exits
- tests prove partial exits do not repeat after state reload
- tests cover remaining quantity and stop update
- commit partial-exit changes

---

## Phase 6: Entry And Exit Review Routing

Split live paper runs into two phases:

- `entry_review`: select symbols, run debate/decision, plan new entries, submit entry orders
- `exit_review`: inspect held positions, apply stop/target/partial/hold-extension logic, submit exit or partial-exit orders

The scheduler should support both phases at separate times. Manual CLI runs should also be able to select a phase so testing does not depend on wall-clock time.

Recommended initial schedule:

- entry review: shortly after market open
- exit review: near market close

Checkpoint:

- phase-routing tests pass
- scheduler tests pass
- commit scheduler and main-routing changes

---

## Phase 7: Paper Dry Run

Before placing paper orders, run with order execution disabled.

Verify:

- correct Alpaca account is detected as paper
- account is empty
- selected strategy is `hold_partial`
- larger-trade approval is disabled
- entry plans look like backtest-style sizing
- exit-review phase handles no-position state cleanly
- reports are written and readable

Checkpoint:

- dry-run report reviewed
- no unexpected live-account or approval behavior
- commit any fixes found during dry run

---

## Phase 8: Paper Launch

Enable paper order execution only after dry-run verification.

First launch should use:

- brand-new paper account
- `hold_partial`
- conditional hold enabled
- partial target exits enabled
- tax-adjusted re-entry disabled
- confidence sizing disabled
- large-trade approval disabled
- conservative account size matching backtest assumptions

During the first week, review every run report and broker response. Do not change strategy parameters mid-week unless there is a safety issue.

Checkpoint:

- first successful entry run
- first successful exit-review run
- live paper report archived
- commit any operational fixes

---

## Acceptance Criteria

- Live paper mode refuses non-paper Alpaca settings.
- Brand-new account guard works.
- Larger-trade Telegram approval is not used in this mode.
- Entry sizing follows backtest-style constraints.
- Conditional thesis-failure hold works and persists across restarts.
- Partial target exits work and persist across restarts.
- Entry and exit-review phases can run independently.
- Dry-run and paper-order reports are written.
- Tests pass before launch.
