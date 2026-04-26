# Confidence Calibration Design

## Goal

Improve the confidence scoring system so a decision's confidence represents the estimated probability that the decision is correct over the next 3 trading days.

## Decision Accuracy Definition

Confidence should mean "probability the system's decision is correct."

The correctness label for each decision is defined over the next 3 trading days:

- `long` is correct when the 3-trading-day move is positive by at least the actionable-move threshold.
- `short` is correct when the 3-trading-day move is negative by at least the actionable-move threshold.
- `skip` is correct when there is no actionable move in either direction over the next 3 trading days.

Because shorting is available in this system, `skip` should not mean "not a profitable long." It should mean that neither a long nor a short would have been actionable.

## Recommended Approach

Use an offline calibration step that learns from prior run artifacts and rescales raw model confidence into observed decision accuracy.

This keeps the current LLM decision flow intact and makes the calibration logic explainable, inspectable, and easy to disable if needed.

## Alternatives Considered

### 1. Inject historical outcomes into the LLM prompt

This could give the model more context, but it would make the behavior harder to audit and more sensitive to prompt shape and token limits.

### 2. Replace model confidence with a fully rules-based score

This would be more deterministic, but it is a much larger change and would throw away useful model signal before we have evidence that a full replacement is better.

## Data Sources

The calibration step should use existing run artifacts plus market data:

- `runs/*/decisions.json`
- `runs/*/selected_symbols.json`
- market-price lookup capable of retrieving the reference price and the close after the next 3 trading days

The historical evaluator should remain read-only with respect to prior runs. It should derive labels from existing artifacts and fetched market data without mutating historical files.

## Labeling Pipeline

For each historical decision:

1. Read the run timestamp and symbol decision.
2. Determine the decision's reference price at the decision date.
3. Find the close after the next 3 trading days.
4. Compute the forward return over that horizon.
5. Convert the forward return into a correctness label using the decision action and the actionable-move threshold.

Rules:

- If a run does not yet have a full 3-trading-day lookahead, exclude it from calibration.
- If price lookup fails for a symbol or date, record the missing evidence and skip that sample.

## Calibration Method

Start with simple bucketed calibration.

Recommended initial buckets:

- `0.60-0.69`
- `0.70-0.79`
- `0.80-0.89`
- `0.90-1.00`

For each bucket:

- count eligible historical decisions
- count correct decisions
- compute observed hit rate

For each new decision:

- place the raw confidence into a bucket
- rescale confidence to the bucket's observed hit rate

## Sparse-Sample Fallback

Do not trust tiny samples.

If a bucket does not have enough examples:

- blend the bucket hit rate with the global hit rate, or
- keep the output confidence closer to the raw value

The exact fallback should be conservative and should avoid overfitting recent noise.

## Integration Point

Keep the current decision engine structure.

The calibration step should happen after the LLM returns `action` and `confidence`, but before downstream sizing and execution logic uses confidence.

This preserves:

- current prompting behavior
- current action generation
- current allocation flow

while improving the meaning of the final confidence score.

## Configuration

Add an explicit configuration value for the actionable-move threshold because `skip` correctness depends on it.

Recommended new setting:

- `CONFIDENCE_ACTIONABLE_MOVE_PCT=0.02`

This means a move must exceed 2% in either direction over the next 3 trading days to count as actionable.

The 3-trading-day horizon should also be explicit in the calibration module, even if it stays fixed initially.

## Reporting And Artifacts

Calibration should emit an artifact that is easy to inspect later.

Recommended outputs:

- bucket ranges
- sample counts per bucket
- correct counts per bucket
- observed hit rate per bucket
- skipped historical samples and why they were skipped
- raw confidence and calibrated confidence for current-run decisions

There should also be one small CLI or reporting surface that summarizes the calibration results without requiring manual JSON inspection.

## Testing

Add focused automated tests for:

- `long` correctness labeling over a 3-trading-day window
- `short` correctness labeling over a 3-trading-day window
- `skip` correctness labeling when neither direction clears the actionable threshold
- bucket calibration behavior
- sparse-sample fallback behavior
- decision-engine integration proving that raw confidence is rescaled before execution uses it

## Non-Goals

This change does not attempt to:

- redesign symbol selection
- replace the LLM decision engine
- optimize for return magnitude instead of directional correctness
- mutate old run artifacts

## Open Assumptions Locked For This Design

These assumptions are explicit to avoid ambiguity during implementation:

- confidence targets decision correctness, not portfolio return
- evaluation horizon is the next 3 trading days
- `skip` means no actionable move in either direction
- shorting is available, so `skip` is judged symmetrically
- initial implementation should favor explainability over sophistication

