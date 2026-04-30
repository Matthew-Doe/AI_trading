from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.utils import get_logger, read_json


def load_symbols_from_snapshot(path: Path) -> list[str]:
    payload = read_json(path)
    rows = payload.get("symbols", payload.get("payload", []))
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a symbols list.")

    symbols = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("symbol"):
            continue
        symbol = str(row["symbol"]).replace(".", "-").upper()
        if symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    if not symbols:
        raise ValueError(f"{path} did not contain any symbols.")
    return symbols


def preload_alpaca_daily_bars(
    service: MarketDataService,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
    batch_size: int,
    feed: str,
) -> dict:
    cached_symbols: list[str] = []
    missing_symbols: list[str] = []

    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        bars_by_symbol = service.fetch_alpaca_daily_bars_bulk(
            batch,
            start=start,
            end=end,
            feed=feed,
        )
        for symbol in batch:
            bars = bars_by_symbol.get(symbol, [])
            if not bars:
                missing_symbols.append(symbol)
                continue
            service.write_daily_bars_cache(symbol, bars)
            cached_symbols.append(symbol)

    return {
        "requested_symbols": len(symbols),
        "cached_symbols": len(cached_symbols),
        "missing_symbols": missing_symbols,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload Alpaca daily bars into local cache.")
    parser.add_argument("--start", required=True, help="Backtest start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Backtest end date, YYYY-MM-DD.")
    parser.add_argument(
        "--symbols-from",
        required=True,
        type=Path,
        help="Universe snapshot JSON containing symbols or payload entries.",
    )
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--feed", default="iex")
    return parser.parse_args()


def _parse_utc_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


def main() -> int:
    args = parse_args()
    config = TradingConfig()
    run_id = f"preload-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    logger = get_logger(config.log_dir, run_id)
    service = MarketDataService(config, logger)

    symbols = load_symbols_from_snapshot(args.symbols_from)
    start = _parse_utc_date(args.start) - timedelta(days=args.lookback_days)
    end = _parse_utc_date(args.end) + timedelta(days=1)

    summary = preload_alpaca_daily_bars(
        service,
        symbols,
        start=start,
        end=end,
        batch_size=args.batch_size,
        feed=args.feed,
    )
    logger.info(
        "Preload complete. requested=%s cached=%s missing=%s",
        summary["requested_symbols"],
        summary["cached_symbols"],
        len(summary["missing_symbols"]),
    )
    if summary["missing_symbols"]:
        logger.warning("Missing symbols: %s", ",".join(summary["missing_symbols"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
