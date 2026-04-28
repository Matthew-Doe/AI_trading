from __future__ import annotations

import argparse
import hashlib
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas_market_calendars as mcal

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.debate import DebateError, OllamaDebateEngine
from trading_system.decision import DecisionEngine, DecisionError
from trading_system.selection import CandidateSelector
from trading_system.utils import ensure_dir, get_logger, write_json, dataclass_to_dict
from trading_system.backtest_execution import BacktestExecutionEngine, BacktestTradeRecord
from trading_system.confidence_calibration import ConfidenceCalibrator
from trading_system.models import TradeDecision

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
            "exit_price_rule": "daily_close_stop_target_or_staged_thesis_exit",
            "sizing_rule": "live_style_risk_allocation_cash_and_position_caps",
            "staged_exit": {
                "min_thesis_days": config.backtest_min_thesis_days,
                "breakeven_after_days": config.backtest_breakeven_after_days,
                "max_hold_days": config.backtest_max_hold_days,
            },
            "tax_rule": "wash_sale_loss_cooldown_and_same_symbol_rebuy_estimate",
            "wash_sale_cooldown_days": execution.wash_sale_cooldown_days,
            "known_limitations": [
                "Universe membership is not point-in-time unless dated universe snapshots were preloaded.",
                "Stops and targets are evaluated on daily close snapshots, not intraday high/low bars.",
                "Short accounting is a conservative cash-reserved approximation, not broker margin simulation.",
                "Tax estimates are same-symbol wash-sale approximations, not broker tax-lot accounting.",
            ],
        },
        "performance": execution.get_summary(),
        "confidence_analysis": build_confidence_analysis(execution.trades),
        "tax": execution.get_tax_summary(),
        "tax_shadow_analysis": execution.get_tax_shadow_analysis(),
        "sizing_logs": execution.sizing_logs,
        "exit_adjustment_logs": execution.exit_adjustment_logs,
        "daily_history": daily_stats,
        "all_trades": [dataclass_to_dict(t) for t in execution.trades],
    }


CONFIDENCE_BUCKETS = (
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
)


def build_confidence_analysis(trades: list[BacktestTradeRecord]) -> dict:
    rows = [_trade_confidence_row(trade) for trade in trades]
    return {
        **_summarize_confidence_rows(rows),
        "buckets": {
            _bucket_label(lower, upper): _summarize_confidence_rows(
                [
                    row
                    for row in rows
                    if row["confidence"] >= lower
                    and (row["confidence"] < upper or (upper == 1.0 and row["confidence"] <= upper))
                ]
            )
            for lower, upper in CONFIDENCE_BUCKETS
        },
        "correlations": {
            "confidence_vs_net_pnl": _correlation(
                [row["confidence"] for row in rows],
                [row["net_pnl"] for row in rows],
            ),
            "confidence_vs_return_pct": _correlation(
                [row["confidence"] for row in rows],
                [row["return_pct"] for row in rows],
            ),
            "confidence_vs_win_loss": _correlation(
                [row["confidence"] for row in rows],
                [1.0 if row["net_pnl"] > 0 else 0.0 for row in rows],
            ),
            "confidence_vs_mfe_pct": _correlation(
                [row["confidence"] for row in rows],
                [row["mfe_pct"] for row in rows],
            ),
            "confidence_vs_mae_pct": _correlation(
                [row["confidence"] for row in rows],
                [row["mae_pct"] for row in rows],
            ),
        },
    }


def _trade_confidence_row(trade: BacktestTradeRecord) -> dict[str, float | str | None]:
    notional = abs(trade.entry_price * trade.qty)
    return_pct = trade.return_pct
    if not return_pct and notional > 0:
        return_pct = trade.net_pnl / notional
    return {
        "confidence": float(getattr(trade, "confidence", 0.0) or 0.0),
        "allocation": float(getattr(trade, "allocation", 0.0) or 0.0),
        "net_pnl": float(trade.net_pnl),
        "return_pct": float(return_pct or 0.0),
        "risk_normalized_return": float(getattr(trade, "risk_normalized_return", 0.0) or 0.0),
        "mfe_pct": float(getattr(trade, "mfe_pct", 0.0) or 0.0),
        "mae_pct": float(getattr(trade, "mae_pct", 0.0) or 0.0),
        "exit_reason": trade.exit_reason,
    }


def _summarize_confidence_rows(rows: list[dict]) -> dict:
    wins = [row for row in rows if row["net_pnl"] > 0]
    losses = [row for row in rows if row["net_pnl"] < 0]
    returns = [row["return_pct"] for row in rows]
    gross_wins = sum(row["net_pnl"] for row in wins)
    gross_losses = -sum(row["net_pnl"] for row in losses)
    return {
        "trade_count": len(rows),
        "win_rate": _ratio(len(wins), len(rows)),
        "average_return_pct": _mean(returns),
        "median_return_pct": _median(returns),
        "average_winner_pct": _mean([row["return_pct"] for row in wins]),
        "average_loser_pct": _mean([row["return_pct"] for row in losses]),
        "profit_factor": round(gross_wins / gross_losses, 4) if gross_losses else None if gross_wins else 0.0,
        "stop_loss_rate": _ratio(
            len([row for row in rows if row["exit_reason"] == "stop_loss"]),
            len(rows),
        ),
        "average_mfe_pct": _mean([row["mfe_pct"] for row in rows]),
        "average_mae_pct": _mean([row["mae_pct"] for row in rows]),
        "average_risk_normalized_return": _mean(
            [row["risk_normalized_return"] for row in rows]
        ),
    }


def _bucket_label(lower: float, upper: float) -> str:
    return f"{lower:.2f}-{upper:.2f}"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_diffs = [value - x_mean for value in xs]
    y_diffs = [value - y_mean for value in ys]
    x_var = sum(value * value for value in x_diffs)
    y_var = sum(value * value for value in y_diffs)
    if x_var <= 0 or y_var <= 0:
        return 0.0
    covariance = sum(x * y for x, y in zip(x_diffs, y_diffs))
    return round(covariance / ((x_var * y_var) ** 0.5), 4)


def build_backtest_skip_decision(symbol: str, reason: str) -> TradeDecision:
    return TradeDecision(
        symbol=symbol,
        action="skip",
        confidence=0.0,
        allocation=0.0,
        catalyst=reason,
    )

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
    
    execution = BacktestExecutionEngine(
        initial_cash=args.initial_cash,
        wash_sale_cooldown_days=config.backtest_wash_sale_cooldown_days,
        config=config,
        market_data_service=market_data,
    )
    
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
            forced_skip_decisions: list[TradeDecision] = []
            debate_failures: list[dict[str, str]] = []
            for symbol_data in selected:
                try:
                    debates.append(debate_engine.run_debate_for_symbol(symbol_data))
                except DebateError as exc:
                    logger.warning(
                        "Skipping %s in backtest after debate JSON failure: %s",
                        symbol_data.symbol,
                        exc,
                    )
                    debate_failures.append(
                        {
                            "symbol": symbol_data.symbol,
                            "error": str(exc),
                        }
                    )
                    forced_skip_decisions.append(
                        build_backtest_skip_decision(
                            symbol_data.symbol,
                            "backtest_debate_failure_default_skip",
                        )
                    )
            write_json(day_path / "debates.json", debates)
            write_json(day_path / "debate_failures.json", debate_failures)

            # 4. Run Decision
            decision_engine = DecisionEngine(config, logger, confidence_calibrator=calibrator)
            if debates:
                try:
                    decisions = decision_engine.decide(debates)
                except DecisionError as exc:
                    logger.warning(
                        "Defaulting %s backtest decisions to skip after decision JSON failure: %s",
                        len(debates),
                        exc,
                    )
                    decisions = [
                        build_backtest_skip_decision(
                            debate.symbol,
                            "backtest_decision_failure_default_skip",
                        )
                        for debate in debates
                    ]
            else:
                decisions = []
            decisions.extend(forced_skip_decisions)
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
