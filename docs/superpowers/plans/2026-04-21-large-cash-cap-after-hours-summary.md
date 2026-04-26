# Large Cash Cap And After-Hours Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the default cash-rich trade cap to 10% and move the default daily summary to 20:00 ET at the end of after-hours trading.

**Architecture:** Keep the change configuration-driven. Update the default values in `trading_system/config.py`, prove the new defaults with focused config tests, and refresh README text that documents the schedule and cash-cap behavior.

**Tech Stack:** Python, pytest, README markdown

---

### Task 1: Lock In New Defaults With Tests

**Files:**
- Modify: `tests/test_config.py`
- Modify: `trading_system/config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trading_config_defaults_cover_cash_rich_cap_and_after_hours_summary():
    config = TradingConfig()

    assert config.cash_rich_trade_pct == 0.10
    assert config.market_close_summary_hour == 20
    assert config.market_close_summary_minute == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk pytest tests/test_config.py::test_trading_config_defaults_cover_cash_rich_cap_and_after_hours_summary -v`
Expected: FAIL because the current defaults are `0.05` and `16:05`.

- [ ] **Step 3: Write minimal implementation**

```python
cash_rich_trade_pct: float = float(os.getenv("CASH_RICH_TRADE_PCT", "0.10"))
market_close_summary_hour: int = int(os.getenv("MARKET_CLOSE_SUMMARY_HOUR_ET", "20"))
market_close_summary_minute: int = int(os.getenv("MARKET_CLOSE_SUMMARY_MINUTE_ET", "0"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk pytest tests/test_config.py::test_trading_config_defaults_cover_cash_rich_cap_and_after_hours_summary -v`
Expected: PASS

### Task 2: Refresh User-Facing Documentation

**Files:**
- Modify: `README.md`
- Test: `rtk pytest tests/test_config.py tests/test_execution.py -q`

- [ ] **Step 1: Update README default summary schedule text**

```md
The market-close Telegram summary uses `BENCHMARK_SYMBOL` and runs at `MARKET_CLOSE_SUMMARY_HOUR_ET:MARKET_CLOSE_SUMMARY_MINUTE_ET`, which defaults to `20:00` for `SPY`.
```

- [ ] **Step 2: Update any README references that still say market close when they mean the scheduled daily summary**

```md
At the end of after-hours trading, the bot can send a daily summary comparing portfolio performance to `SPY`.
```

- [ ] **Step 3: Run focused regression tests**

Run: `rtk pytest tests/test_config.py tests/test_execution.py -q`
Expected: PASS
