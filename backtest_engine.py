from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas_market_calendars as mcal

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.debate import OllamaDebateEngine
from trading_system.decision import DecisionEngine
from trading_system.selection import CandidateSelector
from trading_system.utils import ensure_dir, get_logger, write_json, dataclass_to_dict
from trading_system.backtest_execution import BacktestExecutionEngine, BacktestTradeRecord
from trading_system.confidence_calibration import ConfidenceCalibrator

def get_config_hash(config: TradingConfig) -> str:
    data = f"{config.llm_debate_model}-{config.llm_decision_model}-{config.candidate_count}"
    return hashlib.md5(data.encode()).hexdigest()


def build_backtest_report(
    *,
    config: TradingConfig,
    execution: BacktestExecutionEngine,
    daily_stats: list[dict],
    initial_cash: float,
    start_date: str,
    end_date: str,
    status: str,
    run_at: datetime | None = None,
) -> dict:
    return {
        "metadata": {
            "run_at": (run_at or datetime.now(UTC)).isoformat(),
            "config_hash": get_config_hash(config),
            "initial_cash": initial_cash,
            "days_simulated": len(daily_stats),
            "start_date": start_date,
            "end_date": end_date,
            "status": status,
            "universe_source": "current_companiesmarketcap_snapshot",
            "entry_price_rule": "simulated_open_or_close_with_slippage",
            "exit_price_rule": "daily_close_stop_target_or_time_expiry",
            "known_limitations": [
                "Universe membership is not point-in-time unless dated universe snapshots were preloaded.",
                "Stops and targets are evaluated on daily close snapshots, not intraday high/low bars.",
                "Short accounting is a conservative cash-reserved approximation, not broker margin simulation.",
            ],
        },
        "performance": execution.get_summary(),
        "daily_history": daily_stats,
        "all_trades": [dataclass_to_dict(t) for t in execution.trades],
    }

def run_backtest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-03-01", help="Date to start moving forward from")
    parser.add_argument("--end", default="2026-04-20", help="Date to end simulation")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    args = parser.parse_args()

    config = TradingConfig()
    config.llm_provider = "ollama" 
    
    logger = get_logger(Path("logs"), "backtest")
    market_data = MarketDataService(config, logger)
    selector = CandidateSelector(logger)
    
    execution = BacktestExecutionEngine(initial_cash=args.initial_cash, market_data_service=market_data)
    
    backtest_root = ensure_dir(Path("backtests") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=args.start, end_date=args.end)
    all_days = [d.to_pydatetime().replace(tzinfo=UTC) for d in schedule.index]
    # Simulate OLD-to-NEW (chronological)
    all_days = sorted(all_days)

    print(f"Starting chronological high-fidelity backtest from {args.start} to {args.end}...")
    
    daily_stats = []

    for i, day in enumerate(all_days):
        if i >= args.limit:
            break
            
        print(f"\n[{i+1}/{len(all_days)}] >>> Simulating {day.date().isoformat()} | Equity: ${execution.equity:,.2f}")
        day_path = ensure_dir(backtest_root / day.date().isoformat())
        
        try:
            # 1. Walk-Forward Isolation
            calibrator = ConfidenceCalibrator(
                config=config, 
                logger=logger, 
                market_data_service=market_data,
                run_root=backtest_root,
                now=day,
            )

            # 2. Build Universe
            universe = market_data.build_universe(as_of_date=day)
            selected = selector.select(universe, config.candidate_count)
            write_json(day_path / "selected_symbols.json", selected)
            
            # Map of ALL symbols in universe for stop/exit checks
            full_price_map = {s.symbol: s for s in universe}

            # 3. Run Debates
            debate_engine = OllamaDebateEngine(config, logger)
            debates = []
            for symbol_data in selected:
                debates.append(debate_engine.run_debate_for_symbol(symbol_data))
            write_json(day_path / "debates.json", debates)

            # 4. Run Decision
            decision_engine = DecisionEngine(config, logger, confidence_calibrator=calibrator)
            decisions = decision_engine.decide(debates)
            write_json(day_path / "decisions.json", decisions)

            # 5. Realistic Execution
            execution.process_decisions(decisions, full_price_map, day)
            
            # 6. Record Daily Snapshot
            day_summary = {
                "date": day.date().isoformat(),
                "equity": round(execution.equity, 2),
                "cash": round(execution.cash, 2),
                "open_positions": len(execution.positions),
                "trade_count": len(execution.trades)
            }
            daily_stats.append(day_summary)
            write_json(day_path / "day_summary.json", day_summary)

            # 7. Partial Final Report
            report = build_backtest_report(
                config=config,
                execution=execution,
                daily_stats=daily_stats,
                initial_cash=args.initial_cash,
                start_date=args.start,
                end_date=args.end,
                status="in_progress",
            )
            write_json(backtest_root / "backtest_report.json", report)

        except Exception as e:
            print(f"  Error simulating {day.date().isoformat()}: {e}")
            logger.exception(e)

    # Final Report
    report = build_backtest_report(
        config=config,
        execution=execution,
        daily_stats=daily_stats,
        initial_cash=args.initial_cash,
        start_date=args.start,
        end_date=args.end,
        status="completed",
    )
    write_json(backtest_root / "backtest_report.json", report)
    summary = report["performance"]
    print(f"\nBacktest finished.")
    print(f"Final Equity: ${summary['final_equity']:,.2f}")
    print(f"Net P/L: ${summary['total_net_pnl']:,.2f}")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"Report saved to {backtest_root}/backtest_report.json")

if __name__ == "__main__":
    run_backtest()
