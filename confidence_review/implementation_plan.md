# Implementation Plan for Confidence, Tax, and Exit Improvements

This plan implements the three review tracks in phases so each change can be measured without confusing strategy effects. The first priority is instrumentation and counterfactual tracking. Strategy behavior should change only after the backtest can show whether the new signal is useful.

## Phase 1: Add Measurement Without Changing Behavior

Goal: make the next backtest explain confidence quality, tax opportunity cost, and exit timing.

### Confidence Instrumentation

- Add confidence bucket reporting to the backtest report.
- Track outcomes by raw confidence bucket:
  - trade count
  - win rate
  - average return
  - median return
  - average winner
  - average loser
  - profit factor
  - stop-loss rate
  - average MFE and MAE
- Log normalized outcomes:
  - trade return percentage
  - return per dollar risked
  - return per ATR risked if ATR is available
  - binary win/loss after costs
- Add correlation metrics:
  - confidence vs dollar P/L
  - confidence vs return percentage
  - confidence vs win/loss
  - confidence vs MFE
  - confidence vs MAE

### Tax Instrumentation

- Keep the current tax cooldown behavior unchanged.
- Add shadow tracking for tax-blocked entries.
- For each blocked entry, simulate the trade without affecting cash, positions, or realized P/L.
- Report:
  - blocked entry count
  - blocked entry forward return
  - blocked entry win rate
  - blocked entry MFE and MAE
  - estimated missed P/L
  - estimated tax benefit preserved
- Add symbol-level tax friction reporting:
  - loss exits by symbol
  - blocked re-entries by symbol
  - missed return by symbol

### Exit Instrumentation

- Keep current exit behavior unchanged.
- Split `thesis_failed` into more specific sub-reasons where the data supports it:
  - price invalidation
  - relative strength failure
  - confidence decay
  - max hold / stale thesis
  - opportunity rotation
- Add exit counterfactual tracking:
  - hold 1 more day
  - hold 3 more days
  - hold 5 more days
  - ATR trailing stop if ATR is available
  - partial take-profit plus trailing stop
- Add exit diagnostics:
  - MFE before exit
  - MAE before exit
  - return versus SPY since entry
  - return versus sector or peer proxy if available
  - whether the trade would have recovered after exit

### Phase 1 Acceptance Criteria

- Existing tests pass.
- A small smoke backtest completes.
- Backtest report includes new `confidence_analysis`, `tax_shadow_analysis`, and `exit_counterfactuals` sections.
- No strategy behavior changes are introduced in this phase.

## Phase 2: Improve Confidence Scoring

Goal: stop using raw model confidence as if it were calibrated.

### Confidence Audit Object

Extend decision logging with:

```json
{
  "raw_confidence": 0.74,
  "estimated_win_probability": 0.56,
  "expected_upside_pct": 6.0,
  "expected_downside_pct": 2.5,
  "expected_value_pct": 2.26,
  "risk_reward": 2.4,
  "evidence_count": 4,
  "confidence_cap_reason": null
}
```

### Prompt Changes

- Ask the decision model to estimate upside, downside, win probability, and evidence strength separately.
- Add explicit negative examples for overconfident but weak setups.
- Require confidence to be justified by independent evidence groups.
- Cap confidence when downside is vague or risk/reward is poor.

### Calibration Layer

- Build a simple offline calibration script using historical backtest trades.
- Start with simple models:
  - confidence bucket calibration
  - logistic regression for win/loss
  - isotonic regression for probability calibration
  - optional gradient boosted trees after enough data exists
- Use calibrated confidence for reporting first.
- Only use calibrated confidence for sizing after it beats raw confidence out of sample.

### Phase 2 Acceptance Criteria

- High confidence buckets show monotonic or near-monotonic improvement in win rate, average return, or expected value.
- Calibrated confidence has better correlation with normalized returns than raw confidence.
- Position sizing remains conservative until the calibration signal is proven.

## Phase 3: Improve Tax-Aware Trading

Goal: replace the blunt hard cooldown with tax-adjusted expected value and substitutes.

### Tax Cooldown Tiers

Implement loss-size tiers:

```text
loss <= 0.25%: no cooldown or short cooldown
0.25% < loss <= 1.0%: soft cooldown
loss > 1.0%: full 31-day cooldown
```

Soft cooldown allows re-entry only when calibrated expected value is high enough.

### Tax-Adjusted Expected Value

For blocked symbols, calculate:

```text
tax_adjusted_ev = expected_trade_ev - estimated_tax_penalty - opportunity_cost
```

Allow a re-entry only when tax-adjusted EV clears a strict threshold.

### Substitution Logic

- Build a same-theme substitute list using sector, industry, and universe rank.
- When a symbol is under cooldown, evaluate substitutes before skipping.
- Log the skipped symbol, chosen substitute, and expected-value comparison.

### Lot-Level State

- Add lot-level tracking for backtests first.
- Track:
  - acquisition date
  - quantity
  - cost basis
  - realized gain/loss
  - holding period
  - replacement window
- Defer live lot-level enforcement until backtest behavior is validated.

### Phase 3 Acceptance Criteria

- Tax-blocked entries fall without increasing estimated wash-sale exposure materially.
- Tax-adjusted return improves versus the hard cooldown baseline.
- Shadow blocked-trade data confirms that allowed re-entries or substitutes have positive expected value.

## Phase 4: Improve Exit Logic

Goal: reduce loss churn without cutting off the profitable extended-hold behavior.

### Thesis Failure Confirmation

Replace single-signal thesis failure with confirmation rules:

- price invalidation plus relative weakness
- hold confidence decay plus poor price action
- sector weakness plus symbol underperformance
- large downside breach regardless of other signals

Use staged responses:

- reduce size
- tighten stop
- defer one day
- full exit

### Volatility-Aware Stops

Add ATR-aware stops where data is available:

```text
initial_stop = entry_price - max(min_stop_pct, atr_multiple * ATR)
```

Fallback to existing percent stops when ATR is unavailable.

### Profit Protection

Replace the current breakeven behavior with bands:

```text
after +3% MFE: stop at entry plus estimated costs
after +6% MFE: stop at +2%
after +10% MFE: trail by max(2 ATR, configured percent)
```

### Partial Profit Taking

Test partial exits:

- sell 33% to 50% at first target
- move stop to cost-adjusted breakeven
- trail the remainder
- extend hold while relative strength remains positive

### Phase 4 Acceptance Criteria

- `thesis_failed`, `stop_loss`, and `breakeven_stop` losses decline as a share of gross profit.
- Profit factor improves without relying on fewer trades only.
- Average winner does not collapse.
- Extended-hold contribution remains positive.

## Phase 5: Backtest Sequence

Run the same windows after each phase:

1. Tiny smoke test: 2 to 3 trading days.
2. Short test: Apr 1 to Apr 20.
3. Full comparison: Jan 1 to Apr 20.

For every full comparison, record:

- final equity
- total return
- realized P/L
- win rate
- profit factor
- max drawdown
- average return per trade
- confidence correlations
- confidence bucket monotonicity
- tax-blocked entries
- estimated missed P/L from tax blocks
- exit reason P/L
- exit counterfactual improvement

## Implementation Order

1. Add report-only analytics for confidence buckets, tax shadow entries, and exit counterfactuals.
2. Add tests for report generation and no-behavior-change guarantees.
3. Run a smoke backtest.
4. Run the Apr 1 comparison backtest.
5. Add confidence audit fields and prompt changes.
6. Run calibration analysis from saved backtests.
7. Introduce tax cooldown tiers and tax-adjusted EV.
8. Add substitute selection for tax-blocked symbols.
9. Replace breakeven stop with profit-protection bands.
10. Add partial profit-taking and conditional hold extension.
11. Run the Jan 1 full comparison.
12. Keep only changes that improve risk-adjusted and after-tax results.

## Rollback Rules

Revert or disable a change if it causes any of the following in the Jan-to-Apr comparison:

- lower final equity with higher drawdown
- higher win rate but lower expectancy
- lower tax friction but worse after-tax return
- better raw return caused only by much larger risk
- confidence correlation still near zero while sizing depends more on confidence

The target is not just higher return. The target is a strategy whose confidence, tax behavior, and exit decisions are measurable and predictive enough to trust in live trading.
