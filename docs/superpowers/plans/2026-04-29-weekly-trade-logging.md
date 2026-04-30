# Weekly Trade Logging Implementation Plan

**Goal:** Improve trade logging enough to support weekly strategy reviews, regime adaptation, live paper debugging, and backtest-to-live comparison.

**Core Principle:** Every trade should be auditable from signal to exit. The weekly review should make it easy to decide what to test next without manually digging through raw logs.

---

## Current Gap

Backtest logging is already useful: closed trades include confidence, allocation, return, risk-normalized return, MFE, MAE, sizing reason, and exit reason. Reports also include confidence analysis, tax shadow analysis, exit counterfactuals, sizing logs, and exit adjustment logs.

The missing layer is lifecycle context. A weekly review still needs to connect:

- why a trade was entered
- what regime it was entered in
- which debate and decision fields supported it
- how it was sized
- what order was intended
- what fill actually happened
- what changed while it was held
- why it exited
- whether live paper behaved differently from the backtest

---

## Files To Change

- `trading_system/trade_audit.py`: new audit dataclasses, stable IDs, JSONL read/write helpers, and snapshot builders.
- `trading_system/models.py`: optional IDs or audit context fields where needed.
- `trading_system/backtest_execution.py`: attach trade IDs, decision context, market regime context, and audit events.
- `backtest_engine.py`: pass selected symbols, debates, and decisions into the execution audit context.
- `trading_system/execution.py`: add live paper order lifecycle details to execution results.
- `trading_system/reporting.py`: write audit events alongside live run reports.
- `trading_system/weekly_review.py`: build weekly JSON and markdown summaries.
- `scripts/weekly_trade_review.py`: CLI wrapper for weekly reviews.
- `analyze_backtest_correlation.py`: read the richer confidence and audit fields when present.
- Tests under `tests/`: cover audit primitives, backtest audit logging, live audit logging, and weekly review output.

---

## Phase 1: Canonical Trade Audit Layer

Add a small audit module with:

- stable `trade_id`
- stable `decision_id`
- append-only `AuditEvent`
- decision snapshot structure
- market regime snapshot structure
- JSONL append/load helpers

Use JSONL for lifecycle events because it is simple, append-only, diffable enough for local work, and does not require a database.

Event types should cover:

- decision created
- entry planned
- entry filled
- stop adjusted
- thesis exit deferred
- partial exit filled
- full exit filled
- broker order update
- order rejected
- order canceled

Checkpoint:

- audit serialization tests pass
- stable ID tests pass
- commit audit primitives

---

## Phase 2: Backtest Trade IDs And Lifecycle Context

Every backtest position and closed trade should carry a stable `trade_id`.

Each trade should also carry:

- `decision_id`
- decision snapshot
- compact market snapshot at entry
- regime snapshot at entry
- compact market snapshot at exit when available

Decision snapshots should preserve:

- action
- confidence
- raw confidence
- calibrated confidence
- allocation
- estimated win probability
- expected upside/downside
- expected value
- risk/reward
- evidence count
- confidence cap reason
- target and invalidation prices
- bull and bear confidence
- concise bull and bear arguments and risks

Regime snapshots should classify useful weekly groupings:

- trend regime
- volatility regime
- gap regime
- volume regime
- symbol momentum regime
- simple tags such as overbought, oversold, above/below moving average

Checkpoint:

- backtest trade records include stable IDs
- enriched trades remain backward-compatible JSON
- existing report readers still work
- commit backtest enrichment

---

## Phase 3: Backtest Audit Events

Write `trade_audit_events.jsonl` next to each `backtest_report.json`.

The report should reference the audit event file and include event count metadata, but the main report should not become too large by embedding every lifecycle event.

Backtest events should be emitted for:

- entry fill
- partial target fill
- stop ratchet
- thesis-failure deferral
- full exit
- tax block or tax shadow entry when relevant

Checkpoint:

- audit event file is written during a test backtest
- event count appears in the report metadata
- partial and full exits use the same parent `trade_id`
- commit backtest event logging

---

## Phase 4: Live Paper Broker Lifecycle Logging

Live paper reporting should log broker-facing order details that backtests cannot know.

Capture:

- broker order ID
- client order ID
- submitted timestamp
- side
- quantity
- limit price when used
- stop/take-profit prices when used
- broker status
- filled quantity
- average fill price
- rejection or cancel reason when available
- raw broker response in a nested field for debugging

Write these as audit events so they can be compared with backtest-style planned entries and exits.

Checkpoint:

- dry-run results include the same fields with empty broker values
- paper-order results include broker IDs and fill details
- reporting tests prove audit events are written
- commit live lifecycle logging

---

## Phase 5: Weekly Review Generator

Add a weekly review module and CLI that can read one or more backtest reports and their audit files.

The output should include both:

- machine-readable JSON
- human-readable markdown

The weekly review should summarize:

- total trades
- net P/L
- win rate
- profit factor
- average and median return
- average winner and loser
- max drawdown if available
- trading volume if available
- top winners and losers
- symbol concentration
- side breakdown
- exit-reason breakdown
- confidence bucket breakdown
- regime breakdown
- partial-exit contribution
- conditional-hold contribution
- counterfactual exit findings when available
- tax shadow findings when available

Checkpoint:

- weekly review tests pass
- CLI can generate a review from the January `hold+partial` report
- commit weekly review generator

---

## Phase 6: Adaptation Candidate Detection

The weekly review should not auto-change the strategy. It should identify candidates for controlled A/B tests.

Flag possible review items such as:

- a regime with negative P/L or profit factor below 1
- an exit reason causing repeated losses
- confidence buckets that underperform lower-confidence buckets
- one symbol contributing too much of total profit
- partial exits consistently reducing final P/L
- conditional holds improving or worsening recovery
- live fills materially worse than backtest fills
- repeated broker rejects or partial fills
- JSON repair/retry clusters around losing trades or low-quality decisions

Each candidate should include the evidence and a suggested next experiment, not an automatic parameter change.

Checkpoint:

- tests cover at least regime, exit reason, confidence, and concentration flags
- weekly markdown includes an "Adaptation Candidates" section
- commit adaptation analytics

---

## Phase 7: Validate On January Results

Run the weekly review on the completed January `hold+partial` report.

Review whether it answers:

- what made money
- what lost money
- which exit behavior helped
- which regime was best
- which regime was weakest
- whether confidence was useful
- whether returns were over-concentrated
- what should be A/B tested next

If the review cannot answer one of those questions, add the missing metric before calling the logging upgrade complete.

Checkpoint:

- full tests pass
- January weekly review generated
- example review optionally committed if useful

---

## Acceptance Criteria

- Every new backtest trade has a stable `trade_id`.
- Every enriched trade includes decision, market, and regime context.
- Backtests write append-only audit events.
- Live paper runs write broker lifecycle audit events.
- Weekly review CLI produces markdown and JSON.
- Weekly review includes performance, confidence, exit reason, symbol concentration, and regime breakdowns.
- Weekly review identifies adaptation candidates without changing strategy behavior.
- Existing reports remain readable by existing analysis scripts.
- Full test suite passes.
