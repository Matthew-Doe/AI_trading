# Exploratory January Strategy Results

The January strategy comparisons are exploratory because they were produced before the
bias-safe backtest controls were added. They can be used as implementation references
for strategy behavior, but not as acceptance evidence for live paper trading.

Known local report families:

- baseline January
- conditional hold
- hold + tax
- hold + partial
- hold + tax + partial

The known `hold_tax_partial` local run is `backtests/20260429T225654Z`. Its plan-recorded
summary metrics were:

- final equity: `$107,135.80`
- net P/L: `+$5,026.97`
- win rate: `59.72%`
- profit factor: `2.5552`
- trades: `72`
- top-symbol profit share: `25.34%`

Acceptance-grade reporting now requires point-in-time universe controls, walk-forward
calibration, safer entry timing, conservative intraday exit handling, realistic friction,
and a clean holdout result. Until those controls pass, live paper backtest-style order
execution must fail closed.
