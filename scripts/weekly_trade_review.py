from __future__ import annotations

import argparse
from pathlib import Path

from trading_system.weekly_review import load_reports, write_weekly_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly trade review JSON and markdown.")
    parser.add_argument("reports", nargs="+", type=Path, help="Backtest report JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("confidence_review"))
    parser.add_argument("--stem", default="weekly_trade_review")
    args = parser.parse_args()

    json_path, markdown_path = write_weekly_review(
        load_reports(args.reports),
        output_dir=args.output_dir,
        stem=args.stem,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
