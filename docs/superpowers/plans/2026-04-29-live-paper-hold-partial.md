# Live Paper Hold+Partial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the January-tested `hold+partial` strategy on a brand-new Alpaca paper account with live paper orders that match the backtest timing and behavior as closely as practical.

**Architecture:** Add a paper-only backtest-style live mode that separates morning entry decisions from late-day managed exits. Live entries use the same sizing and planned stop/target math as the backtest, but do not request Telegram approval for larger trades. Live exits are managed by scheduled runs using persisted per-position state so conditional thesis holds and partial target exits survive across process restarts.

**Tech Stack:** Python dataclasses, Alpaca paper trading API, existing scheduler, existing `TradingConfig`, pytest, JSON state files under `runs/`.

---

## Operating Assumptions

- This plan targets a brand-new Alpaca paper account. The implementation must fail closed if the paper account has existing positions or open orders unless explicitly configured otherwise.
- The first live strategy to run is `hold+partial`, not `hold+tax` and not confidence sizing.
- The live mode should behave like the backtest:
  - Morning/open entry decisions.
  - Daily close-style exit review.
  - Stops and targets evaluated by the scheduled exit review, not intraday high/low.
  - Conditional thesis-failure hold extension enabled.
  - Partial target exit enabled.
- This is paper trading only. The implementation must refuse to run with live/non-paper Alpaca settings.
- Do not ask for permission for larger trades. In this mode, the Telegram approval override path must not be used. Sizing should use configured caps directly.
- Do not enable tax-adjusted re-entry in this first live rollout. Tax cooldown blocking can remain in its current live behavior, but `ENABLE_TAX_ADJUSTED_EV_REENTRY` should stay false for this strategy.

## File Structure

- `trading_system/config.py`: add paper-live strategy flags, schedule defaults, and live safety guards.
- `.env.example`: document the exact paper live configuration.
- `trading_system/live_strategy_state.py`: new JSON-backed state store for entry metadata, partial exit status, and thesis deferral counts.
- `trading_system/execution.py`: add live hold+partial sizing, managed exit planning, and no-approval large-trade behavior.
- `trading_system/main.py`: route morning and close-time runs through the paper live strategy mode.
- `trading_system/scheduler.py`: allow separate strategy run times for entry and exit review.
- `tests/test_config.py`: cover config defaults and environment parsing.
- `tests/test_live_strategy_state.py`: cover state persistence and reconciliation.
- `tests/test_execution.py`: cover order plan generation, hold deferrals, partial exits, and no Telegram approval calls.
- `tests/test_main.py` or existing main tests: cover phase routing.

---

### Task 1: Add Paper Live Strategy Config

**Files:**
- Modify: `trading_system/config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add failing config tests**

Add tests that assert:

```python
def test_live_paper_hold_partial_defaults_disabled(monkeypatch):
    for name in (
        "ENABLE_LIVE_PAPER_BACKTEST_STYLE",
        "LIVE_PAPER_STRATEGY",
        "REQUIRE_EMPTY_PAPER_ACCOUNT",
        "ALLOW_LIVE_LARGE_TRADE_APPROVAL",
        "LIVE_ENTRY_REVIEW_TIME_ET",
        "LIVE_EXIT_REVIEW_TIME_ET",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _fresh_trading_config()()

    assert config.enable_live_paper_backtest_style is False
    assert config.live_paper_strategy == "hold_partial"
    assert config.require_empty_paper_account is True
    assert config.allow_live_large_trade_approval is False
    assert config.live_entry_review_time == (9, 31)
    assert config.live_exit_review_time == (15, 55)
```

Add an environment parsing test:

```python
def test_live_paper_hold_partial_config_from_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_PAPER_BACKTEST_STYLE", "true")
    monkeypatch.setenv("LIVE_PAPER_STRATEGY", "hold_partial")
    monkeypatch.setenv("REQUIRE_EMPTY_PAPER_ACCOUNT", "false")
    monkeypatch.setenv("ALLOW_LIVE_LARGE_TRADE_APPROVAL", "false")
    monkeypatch.setenv("LIVE_ENTRY_REVIEW_TIME_ET", "09:35")
    monkeypatch.setenv("LIVE_EXIT_REVIEW_TIME_ET", "15:50")

    config = _fresh_trading_config()()

    assert config.enable_live_paper_backtest_style is True
    assert config.live_paper_strategy == "hold_partial"
    assert config.require_empty_paper_account is False
    assert config.allow_live_large_trade_approval is False
    assert config.live_entry_review_time == (9, 35)
    assert config.live_exit_review_time == (15, 50)
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_config.py -q
```

Expected: fails because the new config fields do not exist.

- [ ] **Step 3: Implement config fields**

Add a helper near `_parse_schedule_times`:

```python
def _parse_hhmm(value: str | None, fallback_hour: int, fallback_minute: int) -> tuple[int, int]:
    if not value or not value.strip():
        return (fallback_hour, fallback_minute)
    hour_text, minute_text = value.strip().split(":", maxsplit=1)
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HH:MM value '{value}'.")
    return (hour, minute)
```

Add fields to `TradingConfig`:

```python
enable_live_paper_backtest_style: bool = _parse_bool_env("ENABLE_LIVE_PAPER_BACKTEST_STYLE")
live_paper_strategy: str = os.getenv("LIVE_PAPER_STRATEGY", "hold_partial").strip().lower()
require_empty_paper_account: bool = _parse_bool_env("REQUIRE_EMPTY_PAPER_ACCOUNT", True)
allow_live_large_trade_approval: bool = _parse_bool_env("ALLOW_LIVE_LARGE_TRADE_APPROVAL", False)
live_entry_review_time: tuple[int, int] = field(
    default_factory=lambda: _parse_hhmm(os.getenv("LIVE_ENTRY_REVIEW_TIME_ET"), 9, 31)
)
live_exit_review_time: tuple[int, int] = field(
    default_factory=lambda: _parse_hhmm(os.getenv("LIVE_EXIT_REVIEW_TIME_ET"), 15, 55)
)
```

- [ ] **Step 4: Update `.env.example`**

Add:

```dotenv
ENABLE_LIVE_PAPER_BACKTEST_STYLE=false
LIVE_PAPER_STRATEGY=hold_partial
REQUIRE_EMPTY_PAPER_ACCOUNT=true
ALLOW_LIVE_LARGE_TRADE_APPROVAL=false
LIVE_ENTRY_REVIEW_TIME_ET=09:31
LIVE_EXIT_REVIEW_TIME_ET=15:55
ENABLE_CONDITIONAL_HOLD_EXTENSION=true
CONDITIONAL_HOLD_EXTENSION_OBSERVATIONS=3
CONDITIONAL_HOLD_MAX_ADVERSE_PCT=0.04
ENABLE_PARTIAL_PROFIT_TAKING=true
PARTIAL_PROFIT_TAKE_FRACTION=0.60
PARTIAL_PROFIT_TRAILING_STOP_PCT=0.03
ENABLE_TAX_ADJUSTED_EV_REENTRY=false
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_config.py -q
```

Commit:

```bash
rtk git add trading_system/config.py .env.example tests/test_config.py
rtk git commit -m "feat: add live paper strategy config"
```

---

### Task 2: Add Persistent Live Strategy State

**Files:**
- Create: `trading_system/live_strategy_state.py`
- Test: `tests/test_live_strategy_state.py`

- [ ] **Step 1: Add state tests**

Create tests for:

```python
def test_live_strategy_state_records_entry_and_partial_exit(tmp_path):
    state = LiveStrategyState(tmp_path / "live_strategy_state.json")
    state.record_entry(
        symbol="AAPL",
        side="long",
        qty=10,
        entry_price=100.0,
        stop_price=94.0,
        take_profit_price=112.0,
        entered_at="2026-04-29T09:31:00-04:00",
    )
    state.record_partial_exit("AAPL", qty_closed=6, remaining_qty=4, exited_at="2026-04-29T15:55:00-04:00")

    reloaded = LiveStrategyState(tmp_path / "live_strategy_state.json")
    position = reloaded.get_position("AAPL")

    assert position["partial_profit_taken"] is True
    assert position["qty"] == 4
    assert position["partial_exit_qty"] == 6
```

```python
def test_live_strategy_state_tracks_thesis_deferrals(tmp_path):
    state = LiveStrategyState(tmp_path / "live_strategy_state.json")
    state.record_entry("AAPL", "long", 10, 100.0, 94.0, 112.0, "2026-04-29T09:31:00-04:00")

    assert state.increment_thesis_deferral("AAPL") == 1
    assert state.increment_thesis_deferral("AAPL") == 2
    state.clear_position("AAPL")
    assert state.get_position("AAPL") is None
```

- [ ] **Step 2: Implement `LiveStrategyState`**

Implement a small JSON-backed class with this shape:

```json
{
  "positions": {
    "AAPL": {
      "symbol": "AAPL",
      "side": "long",
      "qty": 10,
      "entry_price": 100.0,
      "stop_price": 94.0,
      "take_profit_price": 112.0,
      "partial_profit_taken": false,
      "partial_exit_qty": 0,
      "thesis_failure_deferrals": 0,
      "entered_at": "2026-04-29T09:31:00-04:00",
      "updated_at": "2026-04-29T09:31:00-04:00"
    }
  }
}
```

Methods:

```python
class LiveStrategyState:
    def __init__(self, path: Path): ...
    def get_position(self, symbol: str) -> dict | None: ...
    def record_entry(...): ...
    def record_partial_exit(...): ...
    def increment_thesis_deferral(self, symbol: str) -> int: ...
    def reset_thesis_deferral(self, symbol: str) -> None: ...
    def clear_position(self, symbol: str) -> None: ...
    def reconcile_symbols(self, live_symbols: set[str]) -> None: ...
    def snapshot(self) -> dict: ...
```

- [ ] **Step 3: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_live_strategy_state.py -q
```

Commit:

```bash
rtk git add trading_system/live_strategy_state.py tests/test_live_strategy_state.py
rtk git commit -m "feat: add live strategy state"
```

---

### Task 3: Match Backtest-Style Live Sizing and Disable Large-Trade Approval

**Files:**
- Modify: `trading_system/execution.py`
- Test: `tests/test_execution.py`

- [ ] **Step 1: Add execution sizing tests**

Add tests with fake Alpaca account/client and fake Telegram:

```python
def test_live_backtest_style_does_not_request_large_trade_approval(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.allow_live_large_trade_approval = False
    fake_execution.config.high_confidence_threshold = 0.90
    fake_execution.config.high_confidence_trade_pct = 0.10

    plans = fake_execution.build_order_plans(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.95, allocation=0.20)],
        [_market_data("AAPL", close=100.0, atr=2.0)],
    )

    assert plans
    assert plans[0].telegram_approval_required is False
    assert fake_execution.telegram.approval_requests == []
```

Add a test that the plan reason contains enough audit detail:

```python
assert "live_backtest_style=true" in plans[0].reason
assert "approval_disabled=true" in plans[0].reason
```

- [ ] **Step 2: Implement no-approval path**

In `AlpacaExecutionEngine.build_order_plans`, change the high-confidence override branch:

- If `config.enable_live_paper_backtest_style` is false, keep current behavior.
- If it is true and `allow_live_large_trade_approval` is false, do not call `telegram.request_trade_approval`.
- Use the normal max trade cap selected by `_standard_trade_cap`.
- Add `approval_disabled=true` to the order reason.

Do not silently increase max trade size in live paper mode. This satisfies “do not use permission for larger trades” while keeping risk controlled.

- [ ] **Step 3: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_execution.py -q
```

Commit:

```bash
rtk git add trading_system/execution.py tests/test_execution.py
rtk git commit -m "feat: disable live paper large trade approval"
```

---

### Task 4: Implement Conditional Hold for Live Held Positions

**Files:**
- Modify: `trading_system/execution.py`
- Modify: `trading_system/live_strategy_state.py`
- Test: `tests/test_execution.py`

- [ ] **Step 1: Add held-position tests**

Add tests:

```python
def test_live_conditional_hold_defers_losing_position_not_in_active_set(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.enable_conditional_hold_extension = True
    fake_execution.config.conditional_hold_extension_observations = 3
    fake_execution.config.conditional_hold_max_adverse_pct = 0.04
    fake_execution.client.positions = [_position("AAPL", qty=10, market_value=980.0, avg_entry_price=100.0)]

    fake_execution.strategy_state.record_entry("AAPL", "long", 10, 100.0, 94.0, 112.0, "2026-04-29T09:31:00-04:00")

    signals = fake_execution.evaluate_held_positions([], [])

    assert signals[0].signal == "hold"
    assert "Conditional hold deferral" in signals[0].reason
    assert fake_execution.strategy_state.get_position("AAPL")["thesis_failure_deferrals"] == 1
```

```python
def test_live_conditional_hold_sells_after_max_deferrals(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.enable_conditional_hold_extension = True
    fake_execution.config.conditional_hold_extension_observations = 1
    fake_execution.client.positions = [_position("AAPL", qty=10, market_value=980.0, avg_entry_price=100.0)]
    fake_execution.strategy_state.record_entry("AAPL", "long", 10, 100.0, 94.0, 112.0, "2026-04-29T09:31:00-04:00")
    fake_execution.strategy_state.increment_thesis_deferral("AAPL")

    signals = fake_execution.evaluate_held_positions([], [])

    assert signals[0].signal == "sell"
```

```python
def test_live_conditional_hold_sells_when_adverse_move_exceeds_limit(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.enable_conditional_hold_extension = True
    fake_execution.config.conditional_hold_max_adverse_pct = 0.04
    fake_execution.client.positions = [_position("AAPL", qty=10, market_value=940.0, avg_entry_price=100.0)]

    signals = fake_execution.evaluate_held_positions([], [])

    assert signals[0].signal == "sell"
```

- [ ] **Step 2: Implement conditional hold**

In `evaluate_held_positions`, before converting “no active conviction” into `sell`, call a helper:

```python
def _maybe_defer_live_thesis_exit(self, position, side: str, reason: str) -> tuple[bool, str]:
    ...
```

Rules:

- Only active when `ENABLE_LIVE_PAPER_BACKTEST_STYLE=true` and `ENABLE_CONDITIONAL_HOLD_EXTENSION=true`.
- Only defer long positions if unrealized return is negative but greater than `-CONDITIONAL_HOLD_MAX_ADVERSE_PCT`.
- Defer only while state deferral count is less than `CONDITIONAL_HOLD_EXTENSION_OBSERVATIONS`.
- Return `hold` with an audit reason when deferring.
- Sell normally when deferrals are exhausted or adverse move is too large.

- [ ] **Step 3: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_execution.py tests/test_live_strategy_state.py -q
```

Commit:

```bash
rtk git add trading_system/execution.py trading_system/live_strategy_state.py tests/test_execution.py
rtk git commit -m "feat: add live conditional hold exits"
```

---

### Task 5: Implement Live Partial Target Exits

**Files:**
- Modify: `trading_system/execution.py`
- Modify: `trading_system/models.py`
- Modify: `trading_system/live_strategy_state.py`
- Test: `tests/test_execution.py`

- [ ] **Step 1: Extend live order plan reasons without changing API contracts**

Use existing `OrderPlan.side` values when possible:

- `long` / `short` for entries.
- `sell` for exits from long positions.
- `buy_more` for adds.

Do not add a new Alpaca side enum. Add partial-exit audit text in `OrderPlan.reason`, such as:

```text
partial_target_exit=true qty_fraction=0.60 remaining_qty=4
```

- [ ] **Step 2: Add partial target tests**

Add tests:

```python
def test_live_partial_target_creates_fractional_sell_plan(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.enable_partial_profit_taking = True
    fake_execution.config.partial_profit_take_fraction = 0.60
    fake_execution.client.positions = [_position("AAPL", qty=10, market_value=1130.0, avg_entry_price=100.0)]
    fake_execution.strategy_state.record_entry("AAPL", "long", 10, 100.0, 94.0, 112.0, "2026-04-29T09:31:00-04:00")

    signals = fake_execution.evaluate_held_positions(
        [TradeDecision(symbol="AAPL", action="long", confidence=0.8, allocation=0.1)],
        [_market_data("AAPL", close=113.0, atr=2.0)],
    )
    plans = fake_execution.build_held_position_order_plans(signals, [_market_data("AAPL", close=113.0, atr=2.0)])

    assert plans[0].side == "sell"
    assert plans[0].qty == 6
    assert "partial_target_exit=true" in plans[0].reason
```

Add test that no second partial exit is created after `partial_profit_taken=true`.

- [ ] **Step 3: Implement partial target signal**

In `evaluate_held_positions`:

- If live paper mode and partial profit taking are enabled.
- If state has `take_profit_price`.
- If current market data close/latest price is at or above take-profit for long positions.
- If `partial_profit_taken` is false.
- Emit a `HeldPositionSignal` with:
  - `signal="sell"`
  - `target_qty=current_qty - qty_to_close`
  - `delta_qty=-qty_to_close`
  - reason includes `partial_target_exit=true`.

In `build_held_position_order_plans`, create an exit order for `abs(delta_qty)` when `delta_qty < 0`.

After successful order submission, update `LiveStrategyState.record_partial_exit`.

- [ ] **Step 4: Preserve backtest-like close review**

Partial target exits must be generated only during the exit review phase, not during the morning entry phase. The phase routing task controls this.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_execution.py tests/test_live_strategy_state.py -q
```

Commit:

```bash
rtk git add trading_system/execution.py trading_system/models.py trading_system/live_strategy_state.py tests/test_execution.py
rtk git commit -m "feat: add live partial target exits"
```

---

### Task 6: Add Paper Account Safety Guard

**Files:**
- Modify: `trading_system/execution.py`
- Test: `tests/test_execution.py`

- [ ] **Step 1: Add safety tests**

Add tests:

```python
def test_live_paper_mode_requires_paper_base_url(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.alpaca_paper_base_url = "https://api.alpaca.markets"

    with pytest.raises(RuntimeError, match="paper"):
        fake_execution.validate_live_paper_strategy_account()
```

```python
def test_live_paper_mode_requires_empty_account_by_default(fake_execution):
    fake_execution.config.enable_live_paper_backtest_style = True
    fake_execution.config.require_empty_paper_account = True
    fake_execution.client.positions = [_position("AAPL", qty=1, market_value=100.0, avg_entry_price=100.0)]

    with pytest.raises(RuntimeError, match="existing positions"):
        fake_execution.validate_live_paper_strategy_account()
```

- [ ] **Step 2: Implement safety guard**

Add method:

```python
def validate_live_paper_strategy_account(self) -> None:
    if not self.config.enable_live_paper_backtest_style:
        return
    if "paper" not in self.config.alpaca_paper_base_url:
        raise RuntimeError("Live paper strategy mode requires Alpaca paper URL.")
    if self.config.require_empty_paper_account:
        positions = self.client.get_all_positions()
        orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        if positions:
            raise RuntimeError("Live paper strategy mode requires an empty account; existing positions found.")
        if orders:
            raise RuntimeError("Live paper strategy mode requires an empty account; open orders found.")
```

Call this once early in the main live execution flow before order planning.

- [ ] **Step 3: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_execution.py -q
```

Commit:

```bash
rtk git add trading_system/execution.py tests/test_execution.py
rtk git commit -m "feat: guard live paper strategy account"
```

---

### Task 7: Route Entry and Exit Review Phases

**Files:**
- Modify: `trading_system/main.py`
- Modify: `trading_system/scheduler.py`
- Test: `tests/test_main.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Define phases**

Add a pure helper:

```python
def determine_live_paper_phase(config: TradingConfig, now: datetime) -> str:
    local_now = now.astimezone(ZoneInfo(config.market_timezone))
    hour_minute = (local_now.hour, local_now.minute)
    if hour_minute <= config.live_entry_review_time:
        return "entry"
    if hour_minute >= config.live_exit_review_time:
        return "exit_review"
    return "monitor"
```

For the first rollout:

- `entry`: run full universe/debate/decision and submit new entries.
- `exit_review`: run full universe/debate/decision, evaluate held positions, submit partial exits and conditional-hold sells.
- `monitor`: write reports but submit no new orders.

- [ ] **Step 2: Add phase tests**

Test that:

- `09:31 ET` returns `entry`.
- `15:55 ET` returns `exit_review`.
- `12:30 ET` returns `monitor`.

- [ ] **Step 3: Update scheduler**

When live paper mode is enabled, schedule:

```python
config.live_entry_review_time
config.live_exit_review_time
```

Do not also use the broad default `SCHEDULED_TIMES_ET` unless explicitly configured.

- [ ] **Step 4: Update main routing**

In `main.py`:

- During `entry`, call `build_order_plans` for new decisions.
- During `exit_review`, call `evaluate_held_positions` and `build_held_position_order_plans`; do not open new positions unless explicitly configured later.
- During `monitor`, do not submit orders.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
rtk .venv/bin/python -m pytest tests/test_main.py tests/test_scheduler.py -q
```

Commit:

```bash
rtk git add trading_system/main.py trading_system/scheduler.py tests/test_main.py tests/test_scheduler.py
rtk git commit -m "feat: route live paper entry and exit phases"
```

---

### Task 8: Paper Dry-Run Verification

**Files:**
- Use: `trading_system/main.py`
- Use: `runs/*`

- [ ] **Step 1: Run mock mode**

Run:

```bash
rtk .venv/bin/python -m trading_system.main --mock
```

Expected:

- `runs/<run_id>/order_plans.json` exists.
- No Alpaca orders are submitted.
- No live state corruption.

- [ ] **Step 2: Run live paper dry-run with orders disabled**

Use environment:

```bash
ENABLE_LIVE_PAPER_BACKTEST_STYLE=true \
LIVE_PAPER_STRATEGY=hold_partial \
ENABLE_CONDITIONAL_HOLD_EXTENSION=true \
ENABLE_PARTIAL_PROFIT_TAKING=true \
ENABLE_TAX_ADJUSTED_EV_REENTRY=false \
ALLOW_LIVE_LARGE_TRADE_APPROVAL=false \
EXECUTE_ORDERS=false \
rtk .venv/bin/python -m trading_system.main
```

Expected:

- `order_plans.json` is written.
- `execution_results.json` contains `dry_run`.
- No Telegram approval request is made for larger trades.
- `live_strategy_state.json` is created only when simulated fills are explicitly recorded by the implementation.

- [ ] **Step 3: Run full tests**

Run:

```bash
rtk .venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit verification docs**

Create `docs/superpowers/plans/2026-04-29-live-paper-hold-partial-verification.md` with the dry-run commands and observed outputs.

Commit:

```bash
rtk git add docs/superpowers/plans/2026-04-29-live-paper-hold-partial-verification.md
rtk git commit -m "docs: record live paper dry run verification"
```

---

### Task 9: Paper Trading Launch Procedure

**Files:**
- Modify: `.env.example`
- Create: `docs/live-paper-hold-partial-runbook.md`

- [ ] **Step 1: Write launch runbook**

Create `docs/live-paper-hold-partial-runbook.md` with:

```markdown
# Live Paper Hold+Partial Runbook

## Required Settings

ALPACA_API_KEY=<paper key>
ALPACA_SECRET_KEY=<paper secret>
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
EXECUTE_ORDERS=true
ENABLE_LIVE_PAPER_BACKTEST_STYLE=true
LIVE_PAPER_STRATEGY=hold_partial
REQUIRE_EMPTY_PAPER_ACCOUNT=true
ALLOW_LIVE_LARGE_TRADE_APPROVAL=false
ENABLE_CONDITIONAL_HOLD_EXTENSION=true
CONDITIONAL_HOLD_EXTENSION_OBSERVATIONS=3
CONDITIONAL_HOLD_MAX_ADVERSE_PCT=0.04
ENABLE_PARTIAL_PROFIT_TAKING=true
PARTIAL_PROFIT_TAKE_FRACTION=0.60
PARTIAL_PROFIT_TRAILING_STOP_PCT=0.03
ENABLE_TAX_ADJUSTED_EV_REENTRY=false
LIVE_ENTRY_REVIEW_TIME_ET=09:31
LIVE_EXIT_REVIEW_TIME_ET=15:55
ORDER_STYLE=limit
```

## First-Day Procedure

1. Confirm paper account is reset and empty.
2. Run with `EXECUTE_ORDERS=false`.
3. Review `runs/<run_id>/order_plans.json`.
4. Set `EXECUTE_ORDERS=true`.
5. Start scheduler before `09:31 ET`.
6. Confirm paper orders after entry phase.
7. Confirm no Telegram large-trade approval prompts.
8. Confirm exit review runs at `15:55 ET`.

## Daily Review

- Review `execution_results.json`.
- Review `held_position_signals.json`.
- Review `live_strategy_state.json`.
- Compare realized trades against the latest backtest-style metrics.
```

- [ ] **Step 2: Commit runbook**

Run:

```bash
rtk git add .env.example docs/live-paper-hold-partial-runbook.md
rtk git commit -m "docs: add live paper hold partial runbook"
```

---

## Self-Review Notes

- This plan keeps the first paper rollout focused on `hold+partial`, as requested.
- It intentionally excludes confidence sizing because it reduced volume too much.
- It intentionally excludes tax-adjusted re-entry because the request named `hold+partial`, and tax re-entry changes the strategy.
- It makes live timing closer to the backtest by using open-time entries and close-time managed exits.
- It avoids Telegram approval for larger trades by explicitly disabling that live path.
- The largest behavioral mismatch is that live market data is real-time while the backtest uses daily OHLC-derived open/close snapshots. The plan manages that mismatch by limiting strategy exits to the close review phase.
