from __future__ import annotations

import os
import sys
from datetime import date

from trading_system import main as trading_main


DEFAULT_START_DATE = "2026-05-04"
DEFAULT_PHASE = "full"


def _start_date() -> date:
    raw_value = os.getenv("LIVE_PAPER_START_DATE", DEFAULT_START_DATE).strip()
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise SystemExit(f"Invalid LIVE_PAPER_START_DATE={raw_value!r}; expected YYYY-MM-DD.") from exc


def should_run(today: date | None = None) -> bool:
    current_date = today or date.today()
    return current_date >= _start_date()


def _main_args(original_args: list[str]) -> list[str]:
    args = list(original_args)
    if "--phase" not in args:
        args = ["--phase", os.getenv("LIVE_PAPER_PHASE", DEFAULT_PHASE).strip() or DEFAULT_PHASE] + args
    return args


def main() -> int:
    if not should_run():
        print(f"Skipping live paper run before {os.getenv('LIVE_PAPER_START_DATE', DEFAULT_START_DATE)}.")
        return 0

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *_main_args(original_argv[1:])]
        return trading_main.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
