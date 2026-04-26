from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.debate import DebateError, OllamaDebateEngine
from trading_system.decision import DecisionEngine
from trading_system.llm import TokenUsageTracker
from trading_system.execution import AlpacaExecutionEngine
from trading_system.models import (
    DebateResult,
    HeldPositionSignal,
    OrderPlan,
    PendingOrderReview,
    SymbolDebate,
    SymbolMarketData,
    TradeDecision,
)
from trading_system.selection import CandidateSelector
from trading_system.utils import build_run_id, ensure_dir, get_logger, read_json, write_json
from trading_system.telegram import TelegramNotifier
from trading_system.portfolio_summary import send_market_close_summary
from trading_system.reporting import write_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily AI-assisted trading system")
    parser.add_argument("--mock", action="store_true", help="Run with mocked data and skip live APIs.")
    parser.add_argument(
        "--llm-provider",
        choices=("ollama", "openai", "anthropic"),
        help="Override the configured LLM provider.",
    )
    parser.add_argument("--openai-api-key", type=str, help="Override OPENAI_API_KEY for this run.")
    parser.add_argument("--anthropic-api-key", type=str, help="Override ANTHROPIC_API_KEY for this run.")
    parser.add_argument("--llm-debate-model", type=str, help="Override the debate model for this run.")
    parser.add_argument("--llm-decision-model", type=str, help="Override the decision model for this run.")
    parser.add_argument(
        "--include-news", action="store_true", help="Include optional news headlines in prompts."
    )
    parser.add_argument(
        "--replay-run-id",
        type=str,
        help="Replay a previous run from runs/<run_id>/artifacts without rebuilding data.",
    )
    parser.add_argument(
        "--market-close-summary",
        action="store_true",
        help="Send the daily market-close Telegram summary without running the trading pipeline.",
    )
    parser.add_argument("--report-run-id", type=str, help="Generate report files for a prior run.")
    parser.add_argument("--latest-report", action="store_true", help="Generate report files for the latest run.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TradingConfig:
    config = TradingConfig()
    updates: dict[str, object] = {}
    if args.llm_provider:
        updates["llm_provider"] = args.llm_provider
    if args.openai_api_key:
        updates["openai_api_key"] = args.openai_api_key
    if args.anthropic_api_key:
        updates["anthropic_api_key"] = args.anthropic_api_key
    if args.llm_debate_model:
        updates["llm_debate_model"] = args.llm_debate_model
    if args.llm_decision_model:
        updates["llm_decision_model"] = args.llm_decision_model
    return replace(config, **updates) if updates else config


def run_pipeline(config: TradingConfig, args: argparse.Namespace) -> int:
    run_id = build_run_id()
    logger = get_logger(config.log_dir, run_id)
    run_path = ensure_dir(config.run_dir / run_id)
    token_tracker = TokenUsageTracker()
    telegram = TelegramNotifier(config, logger)
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    logger.info("Starting trading run %s", run_id)

    try:
        debate_failures: list[dict[str, str]] = []
        if args.replay_run_id:
            selected_symbols, debates = load_replay_data(config.run_dir / args.replay_run_id)
            universe = selected_symbols
        else:
            market_data = MarketDataService(config, logger)
            if not args.mock and not market_data.is_market_day():
                logger.info("Not a market day. Exiting without trading.")
                return 0

            universe = build_universe(market_data, args.mock, args.include_news)
            selector = CandidateSelector(logger)
            selected_symbols = selector.select(universe, config.candidate_count)
            write_json(run_path / "universe.json", universe)
            write_json(run_path / "selected_symbols.json", selected_symbols)

            debate_engine = OllamaDebateEngine(config, logger, token_tracker=token_tracker)
            if not args.mock:
                debate_engine.warmup()
                debates = []
                for symbol_data in selected_symbols:
                    logger.info("Running sequential debate for %s", symbol_data.symbol)
                    try:
                        debates.append(debate_engine.run_debate_for_symbol(symbol_data))
                    except DebateError as exc:
                        debate_failures.append(
                            {
                                "symbol": symbol_data.symbol,
                                "error": str(exc),
                            }
                        )
                        logger.error(
                            "Skipping %s after debate failure: %s",
                            symbol_data.symbol,
                            exc,
                        )
            else:
                debates = mock_debates(selected_symbols)

            write_json(run_path / "debates.json", debates)
            write_json(run_path / "debate_failures.json", debate_failures)
            if not debates:
                raise RuntimeError("All symbol debates failed; no debates available for decision stage.")

        if args.mock:
            decisions = mock_decisions(debates)
        else:
            decision_engine = DecisionEngine(config, logger, token_tracker=token_tracker)
            decisions = decision_engine.decide(debates)
        write_json(run_path / "decisions.json", decisions)

        order_plans: list[OrderPlan] = []
        held_position_signals: list[HeldPositionSignal] = []
        pending_order_reviews: list[PendingOrderReview] = []
        execution_results: list[dict] = []
        if not args.mock:
            if config.alpaca_api_key and config.alpaca_secret_key:
                execution = AlpacaExecutionEngine(config, logger, run_id=run_id)
                pending_order_reviews = execution.review_pending_orders(universe)
                held_position_signals = execution.evaluate_held_positions(decisions, selected_symbols)
                order_plans = execution.build_held_position_order_plans(
                    held_position_signals, selected_symbols
                ) + execution.build_order_plans(decisions, selected_symbols)
                write_json(run_path / "pending_order_reviews.json", pending_order_reviews)
                write_json(run_path / "held_position_signals.json", held_position_signals)
                write_json(run_path / "order_plans.json", order_plans)
                execution_results = execution.submit_orders(order_plans)
            else:
                logger.warning(
                    "Alpaca credentials not configured. Skipping execution stage after decisioning."
                )
                write_json(run_path / "pending_order_reviews.json", pending_order_reviews)
                write_json(run_path / "held_position_signals.json", held_position_signals)
                write_json(run_path / "order_plans.json", order_plans)
        else:
            order_plans = build_mock_order_plans(decisions, selected_symbols)
            write_json(run_path / "pending_order_reviews.json", pending_order_reviews)
            write_json(run_path / "held_position_signals.json", held_position_signals)
            execution_results = [
                {"symbol": plan.symbol, "status": "mock", "qty": plan.qty, "side": plan.side}
                for plan in order_plans
            ]
            write_json(run_path / "order_plans.json", order_plans)

        write_json(run_path / "execution_results.json", execution_results)
        llm_usage = token_tracker.to_payload()
        write_json(run_path / "llm_usage.json", llm_usage)
        completed_at = datetime.now(UTC)
        elapsed_seconds = perf_counter() - started_perf
        run_metrics = build_run_metrics(started_at, completed_at, elapsed_seconds)
        write_json(run_path / "run_metrics.json", run_metrics)
        write_human_summary(
            run_path,
            selected_symbols,
            debates,
            decisions,
            pending_order_reviews,
            held_position_signals,
            order_plans,
            execution_results,
            llm_usage,
            run_metrics,
        )
        write_run_report(
            run_path=run_path,
            selected_symbols=selected_symbols,
            debates=debates,
            decisions=decisions,
            pending_order_reviews=pending_order_reviews,
            held_position_signals=held_position_signals,
            order_plans=order_plans,
            execution_results=execution_results,
            llm_usage=llm_usage,
            run_metrics=run_metrics,
        )
        print_cli_summary(
            selected_symbols,
            debates,
            decisions,
            pending_order_reviews,
            held_position_signals,
            order_plans,
            execution_results,
            llm_usage,
            run_metrics,
        )
        if telegram.is_enabled():
            telegram.send_message(build_telegram_token_summary(run_id, llm_usage, run_metrics))
        logger.info("Run %s completed successfully", run_id)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Critical failure. Exiting without trading: %s", exc)
        failed_at = datetime.now(UTC)
        elapsed_seconds = perf_counter() - started_perf
        write_json(
            run_path / "failure.json",
            {
                "error": str(exc),
                "timestamp": failed_at.isoformat(),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "started_at": started_at.isoformat(),
                "completed_at": failed_at.isoformat(),
            },
        )
        llm_usage = token_tracker.to_payload()
        run_metrics = build_run_metrics(started_at, failed_at, elapsed_seconds)
        if telegram.is_enabled():
            telegram.send_message(
                f"Run {run_id} failed.\nError: {exc}\n{build_telegram_token_summary(run_id, llm_usage, run_metrics)}"
            )
        return 1


def build_universe(market_data: MarketDataService, mock: bool, include_news: bool) -> list[SymbolMarketData]:
    if mock:
        return load_mock_universe()
    return market_data.build_universe(include_news=include_news)


def load_mock_universe() -> list[SymbolMarketData]:
    from trading_system.models import IndicatorSnapshot, PremarketSnapshot

    universe: list[SymbolMarketData] = []
    mock_symbols = [
        "NVDA",
        "TSLA",
        "AAPL",
        "AMD",
        "META",
        "AMZN",
        "MSFT",
        "PLTR",
        "SMCI",
        "COIN",
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
    ]
    index_proxy_symbols = {"SPY", "QQQ", "DIA", "IWM"}
    for idx, symbol in enumerate(mock_symbols):
        price = 100 + idx * 15
        close = price
        volume = 10_000_000 + idx * 500_000
        indicators = IndicatorSnapshot(
            atr14=close * 0.03,
            rsi14=55 + (idx % 4) * 7,
            sma20=close * 0.99,
            sma50=close * 0.96,
            sma200=close * 0.92,
            volatility20=0.35 + idx * 0.02,
            avg_volume20=volume / 1.4,
        )
        premarket = PremarketSnapshot(
            latest_price=close * 1.01,
            gap_pct=0.01 + idx * 0.002,
            volume=500_000 + idx * 20_000,
            timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),
        )
        universe.append(
            SymbolMarketData(
                symbol=symbol,
                market_cap=None if symbol in index_proxy_symbols else 100_000_000_000 + idx * 1_000_000_000,
                close=close,
                high_20d=price * 1.08,
                low_20d=price * 0.91,
                volume=volume,
                indicators=indicators,
                premarket=premarket,
                price_summary=(
                    f"{symbol} mocked setup with expanding volatility, elevated volume, "
                    f"and premarket continuation gap."
                ),
                news_headlines=[f"{symbol} mock headline {i}" for i in range(1, 3)],
                data_quality_flags=[],
                is_tradeable=True,
                raw_metrics={
                    "price_change_5d": 0.03 + idx * 0.005,
                    "price_change_20d": 0.08 + idx * 0.01,
                    "volume_ratio": 1.4 + idx * 0.1,
                    "range_position": 0.75,
                    "premarket_gap_pct": 0.01 + idx * 0.002,
                    "recent_volatility": 0.35 + idx * 0.02,
                },
            )
        )
    return universe


def mock_debates(selected_symbols: list[SymbolMarketData]) -> list[SymbolDebate]:
    debates: list[SymbolDebate] = []
    for index, symbol_data in enumerate(selected_symbols):
        bull = DebateResult(
            symbol=symbol_data.symbol,
            position="bull",
            confidence=0.60 + index * 0.02,
            arguments=[
                "Momentum and volume confirm expansion.",
                "Price is near resistance with room for breakout.",
            ],
            risks=["Failed breakout", "Macro weakness"],
            key_levels={"support": round(symbol_data.close * 0.97, 2), "resistance": round(symbol_data.close * 1.03, 2)},
            raw_response="mock",
        )
        bear = DebateResult(
            symbol=symbol_data.symbol,
            position="bear",
            confidence=0.45 + index * 0.01,
            arguments=[
                "Move may be crowded and extended.",
                "Gap could fade into open.",
            ],
            risks=["Strong trend persistence", "News catalyst"],
            key_levels={"support": round(symbol_data.close * 0.96, 2), "resistance": round(symbol_data.close * 1.02, 2)},
            raw_response="mock",
        )
        debates.append(SymbolDebate(symbol=symbol_data.symbol, market_data=symbol_data, bull_case=bull, bear_case=bear))
    return debates


def mock_decisions(debates: list[SymbolDebate]) -> list[TradeDecision]:
    weights = [0.20, 0.16, 0.14, 0.12, 0.10]
    decisions: list[TradeDecision] = []
    for idx, debate in enumerate(debates):
        if idx < len(weights):
            decisions.append(
                TradeDecision(
                    symbol=debate.symbol,
                    action="long",
                    confidence=0.70 + (0.02 * max(0, 4 - idx)),
                    allocation=weights[idx],
                )
            )
        else:
            decisions.append(
                TradeDecision(symbol=debate.symbol, action="skip", confidence=0.45, allocation=0.0)
            )
    return decisions


def build_mock_order_plans(decisions: list[TradeDecision], selected_symbols: list[SymbolMarketData]) -> list[OrderPlan]:
    symbol_map = {item.symbol: item for item in selected_symbols}
    plans = []
    capital = 100_000
    for decision in decisions:
        if decision.action == "skip" or decision.allocation <= 0:
            continue
        price = symbol_map[decision.symbol].close
        qty = max(1, int((capital * decision.allocation) / price))
        plans.append(
            OrderPlan(
                symbol=decision.symbol,
                side=decision.action,
                qty=qty,
                notional=qty * price,
                confidence=decision.confidence,
                allocation=decision.allocation,
                reason="mock execution plan",
            )
        )
    return plans


def load_replay_data(run_path: Path) -> tuple[list[SymbolMarketData], list[SymbolDebate]]:
    selected_payload = read_json(run_path / "selected_symbols.json")
    debates_payload = read_json(run_path / "debates.json")
    selected = [symbol_market_data_from_dict(item) for item in selected_payload]
    debates = [symbol_debate_from_dict(item) for item in debates_payload]
    return selected, debates


def symbol_market_data_from_dict(payload: dict) -> SymbolMarketData:
    from trading_system.models import IndicatorSnapshot, PremarketSnapshot

    return SymbolMarketData(
        symbol=payload["symbol"],
        market_cap=payload.get("market_cap"),
        close=payload["close"],
        high_20d=payload["high_20d"],
        low_20d=payload["low_20d"],
        volume=payload["volume"],
        indicators=IndicatorSnapshot(**payload["indicators"]),
        premarket=PremarketSnapshot(**payload["premarket"]),
        price_summary=payload["price_summary"],
        news_headlines=payload.get("news_headlines", []),
        score_breakdown=payload.get("score_breakdown", {}),
        raw_metrics=payload.get("raw_metrics", {}),
        data_quality_flags=payload.get("data_quality_flags", []),
        is_tradeable=payload.get("is_tradeable", True),
    )


def trade_decision_from_dict(payload: dict) -> TradeDecision:
    return TradeDecision(
        symbol=payload["symbol"],
        action=payload["action"],
        confidence=payload["confidence"],
        allocation=payload["allocation"],
        expected_move_pct=payload.get("expected_move_pct"),
        target_price=payload.get("target_price"),
        invalidation_price=payload.get("invalidation_price"),
        time_horizon=payload.get("time_horizon"),
        catalyst=payload.get("catalyst"),
        reward_risk_ratio=payload.get("reward_risk_ratio"),
    )


def pending_order_review_from_dict(payload: dict) -> PendingOrderReview:
    return PendingOrderReview(
        symbol=payload["symbol"],
        order_id=payload.get("order_id"),
        side=payload["side"],
        status=payload["status"],
        submitted_at=payload.get("submitted_at"),
        reference_close=payload.get("reference_close"),
        extended_hours_price=payload.get("extended_hours_price"),
        price_change_pct=payload.get("price_change_pct"),
        action=payload["action"],
        reason=payload["reason"],
    )


def held_position_signal_from_dict(payload: dict) -> HeldPositionSignal:
    return HeldPositionSignal(
        symbol=payload["symbol"],
        current_side=payload["current_side"],
        signal=payload["signal"],
        confidence=payload["confidence"],
        current_qty=payload["current_qty"],
        target_qty=payload["target_qty"],
        delta_qty=payload["delta_qty"],
        reason=payload["reason"],
        max_trade_pct=payload.get("max_trade_pct", 0.0),
    )


def order_plan_from_dict(payload: dict) -> OrderPlan:
    return OrderPlan(
        symbol=payload["symbol"],
        side=payload["side"],
        qty=payload["qty"],
        notional=payload["notional"],
        confidence=payload["confidence"],
        allocation=payload["allocation"],
        reason=payload["reason"],
        max_trade_pct=payload.get("max_trade_pct", 0.0),
        telegram_approval_required=payload.get("telegram_approval_required", False),
        telegram_approval_granted=payload.get("telegram_approval_granted", False),
        entry_limit_price=payload.get("entry_limit_price"),
        stop_price=payload.get("stop_price"),
        take_profit_price=payload.get("take_profit_price"),
        risk_notional=payload.get("risk_notional", 0.0),
        order_style=payload.get("order_style", "market"),
    )


def generate_report_for_run(config: TradingConfig, run_id: str) -> Path:
    run_path = config.run_dir / run_id
    selected_symbols = [
        symbol_market_data_from_dict(item)
        for item in read_json(run_path / "selected_symbols.json")
    ]
    debates = [symbol_debate_from_dict(item) for item in read_json(run_path / "debates.json")]
    decisions = [trade_decision_from_dict(item) for item in read_json(run_path / "decisions.json")]
    pending_order_reviews = [
        pending_order_review_from_dict(item)
        for item in read_json(run_path / "pending_order_reviews.json")
    ]
    held_position_signals = [
        held_position_signal_from_dict(item)
        for item in read_json(run_path / "held_position_signals.json")
    ]
    order_plans = [order_plan_from_dict(item) for item in read_json(run_path / "order_plans.json")]
    execution_results = read_json(run_path / "execution_results.json")
    llm_usage = read_json(run_path / "llm_usage.json")
    run_metrics = read_json(run_path / "run_metrics.json")
    write_run_report(
        run_path=run_path,
        selected_symbols=selected_symbols,
        debates=debates,
        decisions=decisions,
        pending_order_reviews=pending_order_reviews,
        held_position_signals=held_position_signals,
        order_plans=order_plans,
        execution_results=execution_results,
        llm_usage=llm_usage,
        run_metrics=run_metrics,
    )
    return run_path / "report.html"


def generate_report_for_latest_run(config: TradingConfig) -> Path:
    run_dirs = sorted(path for path in config.run_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {config.run_dir}")
    return generate_report_for_run(config, run_dirs[-1].name)


def debate_result_from_dict(payload: dict) -> DebateResult:
    return DebateResult(
        symbol=payload["symbol"],
        position=payload["position"],
        confidence=payload["confidence"],
        arguments=payload["arguments"],
        risks=payload["risks"],
        key_levels=payload["key_levels"],
        raw_response=payload.get("raw_response", ""),
    )


def symbol_debate_from_dict(payload: dict) -> SymbolDebate:
    return SymbolDebate(
        symbol=payload["symbol"],
        market_data=symbol_market_data_from_dict(payload["market_data"]),
        bull_case=debate_result_from_dict(payload["bull_case"]),
        bear_case=debate_result_from_dict(payload["bear_case"]),
    )


def build_run_metrics(started_at: datetime, completed_at: datetime, elapsed_seconds: float) -> dict[str, object]:
    return {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_human": f"{elapsed_seconds:.2f}s",
    }


def build_telegram_token_summary(run_id: str, llm_usage: dict, run_metrics: dict) -> str:
    totals = llm_usage.get("totals", {})
    elapsed_seconds = float(run_metrics.get("elapsed_seconds", 0.0) or 0.0)
    input_tokens = int(totals.get("input_tokens", 0) or 0)
    output_tokens = int(totals.get("output_tokens", 0) or 0)
    total_tokens = int(totals.get("total_tokens", 0) or 0)

    def rate(value: int) -> float:
        return value / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return (
        f"Run {run_id} token summary\n"
        f"Input: {input_tokens} ({rate(input_tokens):.2f} tok/s)\n"
        f"Output: {output_tokens} ({rate(output_tokens):.2f} tok/s)\n"
        f"Total: {total_tokens} ({rate(total_tokens):.2f} tok/s)\n"
        f"Elapsed: {run_metrics.get('elapsed_human')}"
    )


def summarize_execution(order_plans: list[OrderPlan], execution_results: list[dict]) -> list[str]:
    submitted_statuses = {"submitted", "accepted", "new", "partially_filled", "filled", "mock", "dry_run"}
    submitted = [item for item in execution_results if item.get("status") in submitted_statuses]
    skipped = [item for item in execution_results if str(item.get("status", "")).startswith("skipped")]
    lines = [
        (
            f"- planned_orders={len(order_plans)} execution_results={len(execution_results)} "
            f"submitted_or_simulated={len(submitted)} skipped={len(skipped)}"
        )
    ]
    if order_plans and not submitted:
        lines.append("- Execution ran, but no trades were placed.")
    if skipped:
        skip_counts: dict[str, int] = {}
        for item in skipped:
            reason = str(item.get("reason", item.get("status", "skipped")))
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        for reason, count in sorted(skip_counts.items()):
            lines.append(f"- skipped={count} reason={reason}")
    if not order_plans:
        lines.append("- No order plans were generated.")
    return lines


def write_human_summary(
    run_path: Path,
    selected_symbols: list[SymbolMarketData],
    debates: list[SymbolDebate],
    decisions: list[TradeDecision],
    pending_order_reviews: list[PendingOrderReview],
    held_position_signals: list[HeldPositionSignal],
    order_plans: list[OrderPlan],
    execution_results: list[dict],
    llm_usage: dict,
    run_metrics: dict,
) -> None:
    lines = ["Trading Run Summary", f"Generated: {datetime.now(UTC).isoformat()}", ""]
    lines.append("Run Timing")
    lines.append(
        f"- started_at={run_metrics.get('started_at')} completed_at={run_metrics.get('completed_at')} "
        f"elapsed_seconds={run_metrics.get('elapsed_seconds')} elapsed_human={run_metrics.get('elapsed_human')}"
    )
    lines.append("")
    lines.append("Selected Symbols")
    lines.extend(
        f"- {item.symbol}: score={item.score_breakdown.get('total', 0):.2f} summary={item.price_summary}"
        for item in selected_symbols
    )
    lines.append("")
    lines.append("Debates")
    for debate in debates:
        lines.append(
            f"- {debate.symbol}: bull={debate.bull_case.confidence:.2f} bear={debate.bear_case.confidence:.2f}"
        )
    lines.append("")
    lines.append("Decisions")
    lines.extend(
        f"- {item.symbol}: action={item.action} confidence={item.confidence:.2f} allocation={item.allocation:.2f}"
        for item in decisions
    )
    lines.append("")
    lines.append("Pending Order Review")
    lines.extend(
        f"- {item.symbol}: action={item.action} side={item.side} change_pct={(item.price_change_pct or 0):.2%} reason={item.reason}"
        for item in pending_order_reviews
    )
    if not pending_order_reviews:
        lines.append("- No pending orders reviewed.")
    lines.append("")
    lines.append("Held Position Signals")
    lines.extend(
        f"- {item.symbol}: side={item.current_side} signal={item.signal} current_qty={item.current_qty} target_qty={item.target_qty} reason={item.reason}"
        for item in held_position_signals
    )
    lines.append("")
    lines.append("Orders")
    lines.extend(
        f"- {plan.symbol}: side={plan.side} qty={plan.qty} notional={plan.notional:.2f}"
        for plan in order_plans
    )
    if not order_plans:
        lines.append("- No order plans generated.")
    lines.append("")
    lines.append("Execution Summary")
    lines.extend(summarize_execution(order_plans, execution_results))
    lines.append("")
    lines.append("Execution Results")
    lines.extend(f"- {json.dumps(item)}" for item in execution_results)
    if not execution_results:
        lines.append("- No execution results recorded.")
    lines.append("")
    lines.append("LLM Token Usage")
    totals = llm_usage.get("totals", {})
    lines.append(
        f"- total_calls={totals.get('call_count', 0)} input_tokens={totals.get('input_tokens', 0)} "
        f"output_tokens={totals.get('output_tokens', 0)} total_tokens={totals.get('total_tokens', 0)}"
    )
    for stage, stage_totals in llm_usage.get("by_stage", {}).items():
        lines.append(
            f"- stage={stage} calls={stage_totals.get('call_count', 0)} "
            f"input_tokens={stage_totals.get('input_tokens', 0)} "
            f"output_tokens={stage_totals.get('output_tokens', 0)} "
            f"total_tokens={stage_totals.get('total_tokens', 0)}"
        )
    (run_path / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def print_cli_summary(
    selected_symbols: list[SymbolMarketData],
    debates: list[SymbolDebate],
    decisions: list[TradeDecision],
    pending_order_reviews: list[PendingOrderReview],
    held_position_signals: list[HeldPositionSignal],
    order_plans: list[OrderPlan],
    execution_results: list[dict],
    llm_usage: dict,
    run_metrics: dict,
) -> None:
    print(
        "\nRun timing:\n"
        f"  started={run_metrics.get('started_at')} "
        f"completed={run_metrics.get('completed_at')} "
        f"elapsed={run_metrics.get('elapsed_human')}"
    )

    print("\nSelected symbols:")
    for item in selected_symbols:
        print(f"  {item.symbol:>6} score={item.score_breakdown.get('total', 0):.2f}")

    print("\nBull vs Bear:")
    for debate in debates:
        print(
            f"  {debate.symbol:>6} bull={debate.bull_case.confidence:.2f} "
            f"bear={debate.bear_case.confidence:.2f}"
        )

    print("\nFinal decisions:")
    for item in decisions:
        print(
            f"  {item.symbol:>6} action={item.action:<5} "
            f"conf={item.confidence:.2f} alloc={item.allocation:.2f}"
        )

    print("\nPending order review:")
    if pending_order_reviews:
        for item in pending_order_reviews:
            change_pct = item.price_change_pct if item.price_change_pct is not None else 0.0
            print(
                f"  {item.symbol:>6} action={item.action:<6} "
                f"side={item.side:<5} move={change_pct:.2%}"
            )
    else:
        print("  No pending orders reviewed.")

    print("\nHeld positions:")
    if held_position_signals:
        for item in held_position_signals:
            print(
                f"  {item.symbol:>6} signal={item.signal:<8} "
                f"side={item.current_side:<5} qty={item.current_qty}->{item.target_qty}"
            )
    else:
        print("  No held positions evaluated.")

    print("\nOrders:")
    if execution_results:
        for result in execution_results:
            print(f"  {result['symbol']:>6} {result['side']} qty={result['qty']} status={result['status']}")
    else:
        print("  No execution results recorded.")

    print("\nExecution summary:")
    for line in summarize_execution(order_plans, execution_results):
        print(f"  {line.removeprefix('- ')}")

    totals = llm_usage.get("totals", {})
    print("\nLLM token usage:")
    print(
        "  "
        f"calls={totals.get('call_count', 0)} "
        f"input={totals.get('input_tokens', 0)} "
        f"output={totals.get('output_tokens', 0)} "
        f"total={totals.get('total_tokens', 0)}"
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.latest_report:
        report_path = generate_report_for_latest_run(config)
        print(report_path)
        return 0
    if args.report_run_id:
        report_path = generate_report_for_run(config, args.report_run_id)
        print(report_path)
        return 0
    if args.market_close_summary:
        logger = get_logger(config.log_dir, "market_close_summary")
        try:
            send_market_close_summary(config, logger)
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.exception("Market-close summary failed: %s", exc)
            return 1
    return run_pipeline(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
