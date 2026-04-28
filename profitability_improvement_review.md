# Profitability Improvement Review

The current backtests show promise, but the return quality is still fragile. The completed Mar 1 run finished at about +15.3% with a 54.6% win rate, while the current Jan 1 run is around +11.2% through Feb 9 with a 51.0% win rate. Costs are meaningful but not the main issue: reported costs are about 7.6% of gross profits in both runs. The bigger problems are crude exits, weak confidence calibration, repeated same-symbol churn, and differences between live and backtest execution logic.

## Highest Priority Changes

### 1. Replace the fixed time-expiry exit

Most trades are exiting by time rather than by thesis resolution. In the Mar 1 run, 181 of 220 closed trades exited by `time_expiry`; in the Jan 1 run so far, 113 of 145 did. A flat three-day timeout is too blunt because it closes strong but slow setups and holds weak trades until an arbitrary deadline.

Better next test: replace hard expiry with a staged exit rule:

- After 2 trading days, if unrealized P/L is negative and the signal is no longer in the selected set, exit.
- If unrealized P/L is positive but below target, ratchet stop to breakeven or partial-profit level.
- If the trade is still high conviction and trend/volume conditions remain supportive, extend the hold window.
- Log every exit as `thesis_failed`, `breakeven_stop`, `trailing_stop`, `target_hit`, or `extended_hold` instead of collapsing most exits into `time_expiry`.

This should improve both profitability analysis and strategy debuggability.

### 2. Make backtest sizing match live sizing

Live execution uses risk-aware sizing, ATR distance, max trade caps, cash-rich sizing, and position-weight limits. The backtest still behaves more like a simpler allocation/cash simulator. This makes backtest performance less trustworthy because it is not measuring the same risk model that live trading uses.

Next improvement: centralize sizing logic so live and backtest share the same calculation for:

- risk budget per trade
- ATR stop distance
- max position weight
- max total exposure
- cash reserve / cash-rich behavior
- high-confidence sizing behavior

Backtests should report which cap bound each trade: risk, allocation, cash, max position, or tax cooldown.

### 3. Calibrate confidence from realized trade outcomes

The current confidence score has nearly no relationship with trade profit. The completed run had confidence/P&L correlation near zero, and the current run remains weak. The calibrator currently labels synthetic 3-day forward returns as correct/incorrect, but actual trades exit from stop, target, or time logic. That mismatch makes confidence calibration noisy.

Better target: use realized trade outcome data:

- realized net return
- max favorable excursion
- max adverse excursion
- exit reason
- holding period
- whether target was touched before stop

Then score confidence as expected value, not just hit rate. A 45% win-rate setup can still be good if the payoff ratio is high; a 65% hit-rate setup can be bad if losses are too large.

### 4. Add volatility-tier calibration before symbol calibration

Per-symbol calibration sounds attractive, but sample sizes will be too small for most symbols. A more robust first step is volatility-tier calibration:

- low ATR percentage
- medium ATR percentage
- high ATR percentage
- extreme momentum / gap names

This directly addresses the current issue where confidence on a slow mega-cap is treated the same as confidence on a high-volatility momentum stock. Sector calibration can come later once enough trades exist.

### 5. Add tax-aware strategy reporting, not just blocking

The 31-day wash-sale cooldown is useful, but it should be measured as a strategy cost. The report should show:

- trades skipped due to tax cooldown
- estimated P/L missed from skipped trades
- losses deferred by wash-sale prevention
- symbols causing the most tax churn
- raw P/L versus tax-aware estimated P/L

The strategy should eventually decide whether a loss exit is worth triggering a cooldown. Selling a loser is not always bad, but selling and then immediately wanting back in is a signal that the exit logic is probably too noisy.

## Secondary Ideas

### Improve execution assumptions carefully

The current backtest pessimistically applies fixed slippage. Reducing slippage would improve reported returns, but passive limit-fill simulation can become unrealistic fast. Any limit-order backtest needs a fill model:

- fill only if price trades through the limit
- partial-fill assumptions
- missed-trade reporting
- separate maker/taker/slippage estimates

Without that, limit simulation will likely overstate returns.

### Rework stop logic before widening stops

Widening ATR stops may help, but the current backtest evaluates stops on daily close snapshots, not intraday bars. That means the problem is not exactly intraday stop noise. Before changing ATR multiples, add better stop analytics:

- loss by stop distance bucket
- target/stop ratio by volatility tier
- max adverse excursion before profitable exits
- average loss avoided by stop versus time exit

Then test wider stops with risk-normalized sizing so dollar risk per trade stays constant.

### Improve JSON reliability before expanding the schema

Prompting for probability decomposition could improve confidence quality, but the model still emits malformed JSON often. Adding more fields may make this worse. First priority should be reliable structured output. After that, consider adding fields like:

- probability of target before stop
- expected move
- expected downside
- catalyst strength
- technical setup quality
- invalidation clarity

Those fields are useful only if they are consistently valid and later compared against realized outcomes.

## Recommended Next Experiment

The best next branch should not try to optimize everything at once. Run one focused experiment:

1. Make backtest sizing match live sizing.
2. Replace fixed time expiry with staged exit logic.
3. Add richer exit and sizing logs.
4. Keep the universe, prompts, and confidence logic unchanged.
5. Run the same Jan 1 backtest and compare against the current baseline.

Success criteria should be stricter than final return:

- higher profit factor
- lower max drawdown
- fewer arbitrary time exits
- similar or better average trade return
- confidence correlation not worse
- tax churn not worse

If that improves quality, then confidence calibration becomes the next major branch.
