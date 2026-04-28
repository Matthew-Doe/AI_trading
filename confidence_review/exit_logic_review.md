# Exit Logic Review

The current exit logic is doing the right broad thing by letting some winners run, but the loss side appears too mechanical. In the current Jan-start backtest, `thesis_failed`, `stop_loss`, and `breakeven_stop` are responsible for most of the drag. The goal should be to make exits distinguish between normal volatility, thesis invalidation, and opportunity-cost rotation instead of treating every weak mark-to-market move as a reason to leave.

## 1. Split Loss Exits Into Cleaner Categories

`thesis_failed` is too broad to diagnose. It should be broken into specific invalidation reasons:

- price broke the original technical level
- relative strength failed versus SPY
- relative strength failed versus sector or peer group
- volume failed to confirm the move
- catalyst expired or was contradicted
- stop distance became too wide
- better opportunity displaced the position
- model confidence decayed below threshold

This will show whether losses are caused by bad entries, poor hold logic, or normal volatility being misread as failure.

## 2. Add Volatility-Aware Stops

A fixed or shallow stop can punish volatile winners before they have time to work. Stops should be tied to ATR or expected move:

```text
initial_stop = entry_price - max(min_stop_pct, atr_multiple * ATR)
```

The stop should also respect the original thesis. If the thesis requires a breakout level to hold, use that level. If the trade is a mean-reversion entry, use a different structure. One stop style will not fit all setups.

## 3. Require Confirmation Before Thesis Failure

For non-catastrophic losses, thesis failure should require more than one weak signal. For example:

- price below invalidation level
- underperforming SPY since entry
- underperforming sector or peer group
- declining volume quality
- model hold confidence below threshold

Exit immediately only when downside is large or the original thesis is clearly broken. Otherwise, use a staged response: reduce size, tighten stop, or wait one more session for confirmation.

## 4. Replace Breakeven Stop With Profit Protection Bands

The current `breakeven_stop` bucket is still negative after costs and slippage. That means it is functioning as a small-loss exit, not true breakeven protection. Use bands instead:

```text
after +3% MFE: stop at entry plus estimated costs
after +6% MFE: stop at +2%
after +10% MFE: trailing stop at max(close - 2 ATR, entry + 4%)
```

This lets winners breathe while still preventing strong trades from round-tripping into losses.

## 5. Use MFE and MAE to Tune Hold Time

Every exit should be analyzed by maximum favorable excursion and maximum adverse excursion. Key questions:

- Did losers ever become meaningfully profitable first?
- Did winners usually need more than two sessions to work?
- Are target hits leaving too much upside behind?
- Are `thesis_failed` exits happening before normal MAE resolves?

If many losing exits had low MFE and high MAE, entries are bad. If they had high MFE but closed negative, exits are bad. If they had normal MAE before strong gains, stops are too tight.

## 6. Let Winners Run by Scaling Out

Instead of fully exiting at the first target, sell part of the position and trail the rest:

```text
at target 1: sell 33%-50%
move stop to cost-adjusted breakeven
hold remaining shares until trailing stop, thesis failure, or max hold extension
```

This preserves the benefit of high-conviction winners while still realizing gains. It also reduces dependence on picking the perfect take-profit level.

## 7. Make Max Hold Conditional

The current extended hold is profitable, so a hard max hold may be cutting off good trades. Max hold should extend when the trade is still working:

- positive return since entry
- outperforming SPY
- outperforming sector or peers
- no major adverse volume signal
- trailing stop remains above cost-adjusted breakeven

Weak trades can expire quickly. Strong trades should earn more time.

## 8. Add Exit Counterfactuals

For each closed trade, continue tracking what would have happened for several alternative exits:

- hold 1 more day
- hold 3 more days
- hold 5 more days
- ATR trailing stop
- fixed target
- partial target plus trailing stop
- no tax-aware cooldown interaction

This will reveal whether the system exits too early, too late, or at the wrong moments. Counterfactuals are especially useful because they do not require changing live behavior during the first measurement pass.

## 9. Penalize Repeat Loser Patterns

If certain symbols, sectors, or setup types repeatedly hit `thesis_failed` or `stop_loss`, future entries should require stronger evidence. Examples:

- symbol recently stopped out: require higher expected value
- sector has high recent stop rate: reduce size
- model thesis type has poor historical win rate: cap confidence
- repeated small loss exits: force wider initial stop or skip

This keeps the strategy from repeatedly paying for the same bad pattern.

## 10. Improve Exit Logs

Every exit should log a compact decision object:

```json
{
  "exit_reason": "thesis_failed",
  "exit_trigger": "relative_strength_break",
  "entry_confidence": 0.71,
  "hold_confidence": 0.42,
  "return_pct": -2.4,
  "mfe_pct": 3.1,
  "mae_pct": -2.9,
  "return_vs_spy_pct": -1.8,
  "return_vs_sector_pct": -2.2,
  "atr_at_entry": 2.15,
  "atr_multiple_lost": 1.1,
  "tax_cooldown_created": true,
  "would_have_hit_target_next_3d": false
}
```

This makes exit mistakes measurable rather than anecdotal.

## Suggested First Experiment

The first experiment should not change exits immediately. Add exit counterfactual tracking and split `thesis_failed` into specific sub-reasons. Run the same Jan-to-Apr backtest and measure whether current exits beat holding one, three, or five more days. If current exits frequently underperform the counterfactuals, loosen thesis failure and replace breakeven exits with profit-protection bands.
