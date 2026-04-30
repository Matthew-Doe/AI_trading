# Backtest Bias Fixes Implementation Plan

**Goal:** Make the backtest conservative enough that strategy comparisons are useful for live paper deployment, with explicit fixes for future data leakage, survivorship bias, unrealistic entry timing, daily-close exit distortion, and overfitting.

**Core Principle:** Treat the current January results as exploratory. Preserve them as a baseline, then rebuild the backtest so every simulated decision only sees information that would have been available at that time. A strategy should only graduate after it survives a clean holdout window.

---

## Current Risk Summary

The current backtest has several return-inflating risks:

- **Confidence calibration leakage:** prior decisions can be calibrated using future forward returns that were not known yet.
- **Current-universe survivorship bias:** January uses a current top-market-cap universe rather than a point-in-time January tradable universe.
- **Open-price timing leakage:** the day’s official open is used as the simulated premarket price while entries also fill at that open.
- **Future volume leakage:** simulated premarket volume is derived from full-day volume.
- **Daily-close-only exits:** stops, targets, partial exits, and thesis exits are checked only against daily close, missing intraday path risk.
- **Execution friction gap:** fixed slippage does not model spread, latency, partial fills, rejects, short borrow, hard-to-borrow fees, or liquidity impact.
- **January overfitting:** multiple strategy variants were selected on January, so January can no longer be used as an unbiased acceptance window.

---

## Files To Change

- `trading_system/confidence_calibration.py`: enforce time-safe calibration windows.
- `trading_system/data.py`: add point-in-time universe loading, safer historical entry snapshots, and intraday bar access.
- `trading_system/selection.py`: ensure candidate scoring only uses pre-decision fields.
- `trading_system/backtest_execution.py`: add realistic entry timing, intraday stop/target evaluation, and stronger friction modeling.
- `backtest_engine.py`: wire new bias-safe data modes, report bias-control metadata, and support train/holdout runs.
- `trading_system/config.py`: add config flags for bias-safe backtesting modes.
- `analyze_backtest_correlation.py`: show whether a report was run with bias-safe settings.
- `tests/`: add focused tests for leakage prevention, universe snapshots, entry timing, intraday exits, and holdout controls.
- `docs/superpowers/plans/2026-04-29-live-paper-hold-tax-partial.md`: update live-paper readiness criteria after bias-safe holdout results exist.

---

## Phase 0: Preserve Existing Results As Exploratory Baseline

Before changing behavior, record the completed January reports and mark them as exploratory.

Capture:

- baseline January
- conditional hold
- hold + tax
- hold + partial
- hold + tax + partial

Add a short markdown summary under `docs/superpowers/plans/` or `confidence_review/` stating that these results were produced before bias controls and should not be used as final acceptance.

Checkpoint:

- summary references report paths and final metrics
- no behavior changes yet
- commit the summary

---

## Phase 1: Fix Confidence Calibration Leakage

Calibration must never use an outcome whose forward close would not have been known at the current simulated date.

Change the calibration flow so each historical decision outcome has both:

- decision timestamp
- forward outcome timestamp

When calibrating for a simulated day, only include outcomes where the forward outcome timestamp is strictly before the simulated decision time. If the forward result is not yet knowable, exclude it.

Also add a config option to disable calibration entirely for backtest validation runs. The first bias-safe comparison should run both ways:

- calibration off
- calibration on with strict walk-forward outcome availability

Tests should prove:

- a Jan 2 decision with Jan 7 forward outcome cannot calibrate Jan 5
- that same Jan 2 outcome can calibrate dates after Jan 7
- same-day or future outcomes are excluded
- empty calibration history leaves confidence unchanged

Checkpoint:

- calibration leakage tests pass
- existing confidence tests pass
- commit calibration fix

---

## Phase 2: Add Point-In-Time Universe Snapshots

Replace the current top-market-cap universe with dated universe snapshots for backtests.

Preferred first implementation:

- add `data/universe_snapshots/YYYY-MM-DD.json`
- each snapshot contains symbols, names, approximate market caps, and metadata source
- backtest picks the latest snapshot whose date is less than or equal to the simulated day
- if no valid snapshot exists, fail closed unless explicitly configured to allow current-universe fallback

This does not need to be perfect on day one. A coarse monthly snapshot is much better than using today’s winners for a historical month. For January 2026 validation, create or import a snapshot dated before `2026-01-02`.

Report metadata should include:

- universe mode
- universe snapshot date
- symbol count
- whether fallback was used

Tests should prove:

- backtest selects the correct dated snapshot
- current-universe fallback is disabled by default for bias-safe runs
- missing snapshot fails loudly
- index proxy symbols are added only according to config

Checkpoint:

- point-in-time universe tests pass
- January can build a universe from a dated snapshot
- commit universe changes

---

## Phase 3: Remove Open-Price Timing Leakage

The backtest should not both observe the official open and fill at the official open after running expensive selection/debate/decision work.

Add explicit entry timing modes:

- `previous_close_decision_next_open_fill`: decisions use prior close data only, then fill at next open
- `open_plus_delay_fill`: decisions may use opening print/gap data, but fill after a configurable delay using intraday bars or a conservative proxy
- `next_day_open_fill`: safest fallback when intraday data is unavailable

For bias-safe validation, default to one of:

- prior close decision and next open fill
- open plus delay fill with intraday bars

Do not use full-day volume to simulate premarket volume. If true historical premarket data is unavailable, set premarket volume to unknown or a conservative placeholder that cannot influence selection as if it were real.

Tests should prove:

- decision features exclude same-day open in prior-close mode
- same-day open can only be used in delayed-fill mode
- fill price is not identical to the decision input price when delay mode is active
- premarket volume is not derived from full-day volume in bias-safe mode

Checkpoint:

- entry timing tests pass
- report metadata states entry timing mode and fill rule
- commit timing changes

---

## Phase 4: Add Intraday Stop, Target, And Partial Exit Simulation

Daily close is not enough for a stop/target strategy. Add intraday path simulation for exits.

Preferred implementation:

- fetch historical minute or 5-minute bars for open positions
- evaluate stop, target, partial target, and trailing stop in chronological bar order
- apply conservative ordering when both stop and target occur inside the same bar
- record whether an exit was intraday or close-based

Fallback when intraday bars are missing:

- use daily high/low to detect possible stop/target hits
- if both stop and target were touched, choose the worse outcome first
- mark the trade with an `intraday_data_missing` flag

Tests should prove:

- stop hit before target exits as stop
- target hit before stop exits as target or partial target
- same-bar ambiguity resolves conservatively
- partial target updates remaining quantity and trailing stop before later bars are evaluated
- missing intraday data is flagged

Checkpoint:

- intraday exit tests pass
- January smoke backtest runs with intraday or conservative daily high/low fallback
- commit intraday exit simulation

---

## Phase 5: Improve Execution Friction Modeling

Add configurable execution friction that is closer to live paper/live trading.

Model at least:

- bid/ask spread estimate
- slippage that grows with volatility and gap size
- liquidity cap based on a small percentage of bar volume
- partial fills when desired quantity exceeds liquidity cap
- no short entry when borrow is unavailable or symbol is not shortable
- optional hard-to-borrow fee estimate for shorts
- order reject/cancel simulation for impossible limit prices

This does not need to become a broker simulator. The goal is to stop assuming every selected trade fills instantly at a favorable price.

Tests should prove:

- high-volume names fill normally
- low-volume names are capped or partially filled
- wider spread increases cost
- shorts can be blocked by shortability config
- report includes total friction cost and skipped/partial-fill counts

Checkpoint:

- friction tests pass
- report metadata includes friction model settings
- commit execution friction changes

---

## Phase 6: Add Bias-Control Metadata And Report Gates

Every backtest report should make bias controls visible.

Add a `bias_controls` section to `backtest_report.json`:

- point-in-time universe enabled
- universe snapshot date
- calibration mode
- maximum calibration outcome timestamp used
- entry timing mode
- intraday exit mode
- friction model version
- current-universe fallback used
- missing intraday bar count
- stale cache count

Add warning flags when a run is not acceptance-grade:

- current universe used
- same-day open used for immediate fill
- daily-close-only exits
- calibration outcomes not strictly walk-forward
- intraday data missing for too many trades

Tests should prove:

- old reports still load
- new reports include bias controls
- acceptance-grade flag is false when any hard bias control is missing

Checkpoint:

- reporting tests pass
- analysis script prints bias-control status
- commit reporting changes

---

## Phase 7: Establish Train/Holdout Protocol

January has been used for strategy selection, so it should become the training/exploration month.

Use this protocol going forward:

- January 2026: strategy exploration and parameter tuning
- February 2026: primary holdout acceptance
- March 2026: secondary holdout if February passes
- April remains excluded from training and acceptance unless explicitly designated later

Acceptance should require improvement over baseline on:

- net P/L
- profit factor
- max drawdown
- median return
- concentration
- trading volume or at least no unacceptable volume collapse
- after-tax estimate where tax behavior is part of the strategy

Do not keep a strategy that only improves headline return by increasing concentration, reducing volume too far, or relying on one symbol.

Checkpoint:

- backtest CLI supports named train/holdout windows or documented commands
- holdout summary template exists
- commit protocol docs

---

## Phase 8: Re-Run Strategy Comparison Under Bias-Safe Settings

After the fixes, rerun controlled comparisons.

Order:

1. January baseline under bias-safe settings
2. January `hold+tax+partial` under bias-safe settings
3. February baseline under bias-safe settings
4. February `hold+tax+partial` under bias-safe settings
5. Optional March holdout only if February passes

Do not change parameters after seeing February. If February fails, go back to January exploration, make a new candidate, then test on a different untouched holdout.

Checkpoint:

- all runs complete
- weekly review generated for each candidate and holdout
- live-paper plan updated with final accepted strategy or downgraded to dry-run only
- commit summaries, not massive raw artifacts unless intentionally tracked

---

## Acceptance Criteria

- Confidence calibration cannot use unknowable future outcomes.
- Backtests can run with point-in-time universe snapshots and fail closed without them.
- Bias-safe entry timing does not use the official open and fill at that same open after decisions.
- Stops, targets, partial exits, and trailing stops are evaluated with intraday bars or conservative high/low fallback.
- Execution friction includes spread/slippage/liquidity constraints.
- Reports clearly state whether the run is acceptance-grade.
- January results are marked exploratory.
- `hold+tax+partial` is only approved for live paper after passing a clean holdout.
- Full unit suite passes after each implementation checkpoint.

---

## Priority Order

If time is limited, fix in this order:

1. Confidence calibration leakage.
2. Point-in-time universe snapshots.
3. Entry timing leakage.
4. Intraday or conservative stop/target simulation.
5. Execution friction.
6. Holdout automation and reporting.

The first three are hard blockers for trusting the backtest. The rest improve realism and reduce the chance that a live paper deployment looks much worse than the simulated result.
