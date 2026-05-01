# Live Paper Daily Rollout Design

## Goal

Run the `implement-all-three-plans` branch once per weekday starting May 4, 2026, first validating with the current Alpaca paper account and then switching to a fresh Alpaca paper account for the scheduled run.

## Architecture

Use a user-level systemd timer for the once-daily schedule instead of the app's long-running APScheduler process. The timer invokes a small Python wrapper that checks the configured start date before calling `trading_system.main` with `--phase full`.

Keep account credentials out of git. The checked-in files provide the service, timer, start-date guarded runner, and an operations runbook; `.env` remains local and ignored.

## Runtime Behavior

- The scheduled service runs from the feature-branch worktree.
- `LIVE_PAPER_START_DATE=2026-05-04` prevents accidental runs before Monday, May 4, 2026.
- `LIVE_PAPER_DAILY_TIME_ET=09:45` is documented as the default timer time.
- Validation uses the current account with `EXECUTE_ORDERS=false`.
- May 4 production paper use swaps in the fresh Alpaca paper credentials and may set `EXECUTE_ORDERS=true` after validation artifacts are reviewed.

## Error Handling

The wrapper exits successfully before the start date so the timer does not enter a failed state. On or after the start date, the normal pipeline exit code is preserved so systemd records failures.

## Testing

Tests cover the date guard and argument forwarding behavior with the real wrapper entry point. A syntax/import check verifies the script can be loaded in the project environment.
