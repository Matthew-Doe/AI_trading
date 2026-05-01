# Live Paper Daily Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe once-daily live-paper rollout setup for the `implement-all-three-plans` branch starting May 4, 2026.

**Architecture:** Add a small Python wrapper around `trading_system.main` that enforces `LIVE_PAPER_START_DATE` before running `--phase full`. Add systemd user service/timer files and an operations runbook for current-account validation and fresh-account cutover.

**Tech Stack:** Python 3.11+, pytest, systemd user timers, existing `TradingConfig`/dotenv configuration.

---

### Task 1: Start-Date Guarded Runner

**Files:**
- Create: `scripts/run_live_paper_daily.py`
- Test: `tests/test_run_live_paper_daily.py`

- [ ] **Step 1: Write tests for before-date skip and on-date forwarding**

Use pytest to monkeypatch `date.today`, environment variables, and `trading_system.main.main`.

- [ ] **Step 2: Implement the runner**

The runner parses `LIVE_PAPER_START_DATE`, skips before the date, sets `sys.argv` to run `--phase full`, and returns the wrapped main exit code.

- [ ] **Step 3: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_run_live_paper_daily.py -q`

### Task 2: systemd User Timer

**Files:**
- Create: `deploy/systemd/ai-trading-live-paper-daily.service`
- Create: `deploy/systemd/ai-trading-live-paper-daily.timer`

- [ ] **Step 1: Add service file**

The service uses the feature worktree as `WorkingDirectory`, loads `.env`, and runs `.venv/bin/python scripts/run_live_paper_daily.py`.

- [ ] **Step 2: Add timer file**

The timer runs weekdays at 09:45 America/New_York-equivalent local timer time with persistence enabled.

### Task 3: Operations Runbook

**Files:**
- Create: `docs/live-paper-daily-rollout.md`

- [ ] **Step 1: Document validation**

Copy current `.env` locally, run with `EXECUTE_ORDERS=false`, and inspect `runs/<run_id>/order_plans.json` and `execution_results.json`.

- [ ] **Step 2: Document May 4 cutover**

Replace Alpaca credentials with the fresh paper account, enable the user timer, and only set `EXECUTE_ORDERS=true` after validation.

### Task 4: Verification and Checkpoint

**Files:**
- Modify only the files above.

- [ ] **Step 1: Run tests**

Run: `.venv/bin/python -m pytest tests/test_run_live_paper_daily.py tests/test_scheduler.py tests/test_config.py -q`

- [ ] **Step 2: Validate current account**

Run one current-account validation with `EXECUTE_ORDERS=false`.

- [ ] **Step 3: Commit**

Commit the setup files after tests and validation.
