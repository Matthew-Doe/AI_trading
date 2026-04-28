# Strategies to Improve Confidence Correlation

The current confidence score appears weakly correlated with realized profit. That suggests the score is acting more like a narrative certainty estimate than a calibrated probability of trade success. The goal should be to make confidence mean something measurable: higher confidence should imply better expected return, better win rate, lower drawdown, or some explicit combination of those outcomes.

## 1. Redefine Confidence as Expected Value

The system should stop treating confidence as a general conviction score. Instead, define it as expected trade value:

```text
confidence = probability_of_win * expected_winner_size - probability_of_loss * expected_loser_size
```

This forces confidence to account for asymmetric payoff. A trade with a 70% win probability but tiny upside should not outrank a 52% win probability trade with strong upside and tight invalidation. Backtest logs should store the model's estimated win probability, expected upside, expected downside, and resulting expected value separately.

## 2. Train a Calibration Layer on Backtest Outcomes

Use historical backtest trades to build a simple calibration model on top of the LLM output. Inputs can include model confidence, universe rank, momentum score, volatility, sector, market regime, recent gap size, distance from moving averages, ATR risk, liquidity, and prior symbol behavior. The output should predict one or more concrete labels:

- positive net P/L
- return above fees/slippage
- return above a minimum hurdle, such as 2%
- stop-loss probability
- expected return per dollar risked

Start with logistic regression, isotonic regression, or gradient boosted trees before using anything more complex. The goal is not sophistication; it is whether the raw confidence score can be corrected into a calibrated ranking signal.

## 3. Penalize Unsupported Confidence

High confidence should require agreement between independent evidence groups. For example:

- price momentum confirms the thesis
- volume or relative strength confirms demand
- volatility supports the stop distance
- sector or market breadth is not hostile
- news/catalyst thesis is specific rather than generic
- downside estimate is explicit and bounded

If the model gives high confidence without enough evidence groups, cap the confidence. This should reduce overconfident losing trades where the explanation sounds good but the setup is thin.

## 4. Separate Entry Confidence From Exit Confidence

The current score may be mixing two different questions:

- Should we enter this trade today?
- Should we keep holding this trade?

Those should be logged and calibrated separately. Entry confidence should predict forward return from a fresh position. Hold confidence should predict whether the remaining trade has positive expected value from the current price. Many `thesis_failed` exits suggest the system may be entering before the thesis is durable enough or failing to update the thesis consistently after entry.

## 5. Add Confidence Decay and Confirmation Rules

Confidence should decay when the trade does not move as expected after entry. For example, a high-confidence trade that fails to outperform SPY, sector ETF, or its original momentum basket after two sessions should lose confidence even if price has not hit a stop. Conversely, confidence should increase only when the market confirms the expected path.

Useful confirmation metrics:

- return versus SPY since entry
- return versus sector or peer basket since entry
- intraday or daily close above entry thesis level
- whether max adverse excursion stayed within the forecast risk band
- whether volume increased on favorable moves

## 6. Score Confidence by Risk-Adjusted Outcome

Profit-only correlation can be noisy because larger positions produce larger dollar P/L. Measure confidence correlation against normalized outcomes too:

- trade return percentage
- return per ATR risked
- return divided by max adverse excursion
- return divided by planned downside
- binary win/loss after costs
- whether the trade hit target before stop

If confidence correlates with dollar P/L but not normalized returns, the sizing system is doing the work. If it correlates with normalized returns, the signal is actually useful.

## 7. Create Confidence Buckets and Enforce Monotonicity

Every backtest should report outcome by confidence bucket:

```text
0.50-0.60
0.60-0.70
0.70-0.80
0.80-0.90
0.90-1.00
```

Each bucket should show trade count, win rate, average return, median return, average winner, average loser, profit factor, drawdown contribution, and stop-loss rate. If higher confidence buckets do not produce better outcomes, do not use confidence for sizing until it is fixed.

## 8. Add Negative Examples to the Prompt

The decision prompt should include examples of setups that sound attractive but should receive low confidence. Examples:

- strong recent momentum after an exhausted multi-day move
- high-beta stock moving only because the whole market is up
- rebound candidate below declining moving averages
- large upside story with vague stop-loss logic
- high confidence despite poor risk/reward
- catalyst already priced in

This can reduce inflated confidence on familiar but low-quality narratives.

## 9. Log the Raw Evidence Behind Confidence

For each decision, log a compact confidence audit object:

```json
{
  "raw_confidence": 0.82,
  "calibrated_confidence": 0.61,
  "estimated_win_probability": 0.55,
  "expected_upside_pct": 6.0,
  "expected_downside_pct": 2.5,
  "expected_value_pct": 2.18,
  "evidence_count": 4,
  "risk_reward": 2.4,
  "market_alignment": "neutral",
  "confidence_cap_reason": null
}
```

This makes it possible to debug whether bad confidence comes from the model, the market data, the prompt, or the sizing layer.

## 10. Gate Position Size Until Confidence Is Proven

Until confidence correlation improves, confidence should not drive aggressive sizing directly. Use small position-size differences between buckets, or require calibrated confidence plus independent technical confirmation before increasing allocation. This prevents a bad score from amplifying losses while still allowing the backtest to collect calibration data.

## Suggested First Experiment

The first practical experiment should be small and measurable: add confidence bucket reporting, log expected upside/downside separately, and compute correlation against normalized trade return, return per ATR risk, and binary win/loss. Then run the same backtest window and compare whether the highest confidence bucket actually beats the middle buckets. If it does not, cap confidence-based sizing and start building a calibration layer from the logged features.
