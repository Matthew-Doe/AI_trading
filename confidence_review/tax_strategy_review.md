# Tax-Aware Strategy Review

The current tax-aware logic is useful as a first guardrail, but it is blunt. A 31-day same-symbol cooldown after any realized loss avoids obvious wash-sale mistakes, yet it can also suppress valid recovery trades and leave too much cash idle. The tax layer should be treated as an optimizer, not only a blocker: it should compare expected trade value against tax risk, defer weak trades, and allow strong substitutes when a direct rebuy would be costly.

## 1. Separate Tax Avoidance From Trade Rejection

A wash-sale risk should not automatically mean the strategy has no trade. Instead, the system should choose among:

- skip the trade
- buy a correlated but not substantially identical substitute
- wait until cooldown expires
- use a smaller size if expected value justifies the tax risk
- allow the trade only if it is already high-confidence and high expected value

Backtests should log which alternative was chosen and what expected return was sacrificed. That will show whether tax protection is saving money or simply blocking winners.

## 2. Add a Tax Opportunity-Cost Metric

The current report counts tax-blocked entries, but the more important question is what those blocked trades would have done. For each blocked entry, the backtest should continue tracking a virtual shadow trade with no capital impact.

Useful metrics:

- blocked trade count
- blocked trade average forward return
- blocked trade win rate
- blocked trade maximum favorable excursion
- blocked trade maximum adverse excursion
- estimated P/L missed by tax cooldown
- estimated tax benefit preserved

If blocked trades are profitable after costs more often than not, the cooldown is too aggressive. If they are mostly losers, the tax layer is also acting as a useful risk filter.

## 3. Rank Re-Entries by Tax-Adjusted Expected Value

When a symbol is under cooldown, compute a tax-adjusted expected value:

```text
tax_adjusted_ev = expected_trade_ev - estimated_tax_penalty - opportunity_cost_of_capital
```

The model should only override cooldown when the tax-adjusted expected value remains strongly positive. This is better than treating all rebuy attempts equally. A weak rebound attempt should stay blocked, but a rare high-quality continuation setup should not be treated the same way.

## 4. Prefer Substitution Baskets

For symbols under cooldown, the strategy should search for substitutes in the same theme without buying something clearly identical. Examples:

- semiconductor loser cooldown: consider another semiconductor or equipment name
- software loser cooldown: consider a different high-relative-strength software stock
- index-like exposure cooldown: use a broader market or sector substitute carefully

The substitute should pass its own entry criteria. It should not be a mechanical replacement. The point is to preserve exposure to the thesis while reducing wash-sale risk.

## 5. Stop Harvesting Weak Losses Too Quickly

Some `thesis_failed` exits may be creating tax cooldowns too early. If the strategy sells a position at a small loss after one or two sessions and then wants to rebuy soon after, the original exit may have been premature. The tax system should feed back into exit logic:

- avoid selling at a small loss if the thesis is still valid
- prefer reducing size over full exit when wash-sale cooldown would likely block a good re-entry
- require stronger invalidation before creating a tax-loss cooldown
- allow full exit when downside risk is large enough to dominate tax concerns

This turns tax awareness into a trade-management input instead of a post-sale constraint.

## 6. Track Lot-Level Tax State

The current cooldown approach is symbol-level. A more accurate version should track lots:

- acquisition date
- quantity
- cost basis
- realized gain or loss per lot
- holding period
- replacement-share window
- whether any loss is potentially disallowed

Lot-level tracking matters when only part of a position is sold, when a symbol has multiple buys, or when add-on trades occur near loss exits. It also makes the system ready for live reconciliation.

## 7. Add Short-Term Holding Period Awareness

Since this strategy trades frequently, most gains will likely be short-term. The tax layer should estimate after-tax expected value using a configurable marginal tax rate. Even a profitable trade can be unattractive after short-term taxes if the edge is small and turnover is high.

Backtest reports should include:

- estimated short-term gains
- estimated short-term losses
- estimated net taxable short-term gain
- estimated tax due
- after-tax return estimate
- after-tax profit factor
- tax drag as percent of gross profit

This keeps optimization focused on spendable returns, not just pre-tax P/L.

## 8. Tune Cooldown by Loss Size

Not every loss deserves the same cooldown response. A tiny loss caused by slippage or fees should not block the same way as a real thesis failure. Suggested tiers:

```text
loss <= 0.25%: no cooldown or short cooldown
0.25% < loss <= 1.0%: soft cooldown requiring high expected value
loss > 1.0%: full 31-day cooldown unless substitute is used
```

This should reduce unnecessary blocking while still avoiding the obvious wash-sale traps.

## 9. Add Tax-Aware Reporting by Symbol

For each symbol, report:

- realized gain
- realized loss
- number of loss exits
- number of blocked re-entries
- missed return from blocked re-entries
- substitute trades taken
- estimated wash-sale risk avoided

This will identify whether a few high-turnover names are responsible for most tax friction. If so, the fix may be symbol-level rules, not a global tax policy.

## Suggested First Experiment

The first test should add shadow tracking for blocked tax entries. Keep the actual cooldown behavior unchanged, but record what would have happened if each blocked trade had been allowed. After one full Jan-to-Apr backtest, compare estimated tax benefit preserved against estimated missed P/L. If missed P/L is larger than avoided tax risk, replace the hard cooldown with a tax-adjusted expected-value gate.
