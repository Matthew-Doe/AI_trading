from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from trading_system.models import (
    HeldPositionSignal,
    OrderPlan,
    PendingOrderReview,
    SymbolDebate,
    SymbolMarketData,
    TradeDecision,
)
from trading_system.utils import dataclass_to_dict, write_json


def build_run_report_payload(
    *,
    selected_symbols: list[SymbolMarketData],
    debates: list[SymbolDebate],
    decisions: list[TradeDecision],
    pending_order_reviews: list[PendingOrderReview],
    held_position_signals: list[HeldPositionSignal],
    order_plans: list[OrderPlan],
    execution_results: list[dict],
    llm_usage: dict,
    run_metrics: dict,
) -> dict[str, Any]:
    data_quality_warnings = [
        {
            "symbol": item.symbol,
            "flags": item.data_quality_flags,
            "is_tradeable": item.is_tradeable,
        }
        for item in selected_symbols
        if item.data_quality_flags or not item.is_tradeable
    ]
    return {
        "summary": {
            "selected_symbols": len(selected_symbols),
            "debates": len(debates),
            "active_decisions": len([item for item in decisions if item.action != "skip"]),
            "planned_orders": len(order_plans),
            "execution_results": len(execution_results),
            "elapsed_human": run_metrics.get("elapsed_human"),
        },
        "data_quality_warnings": data_quality_warnings,
        "selected_symbols": [dataclass_to_dict(item) for item in selected_symbols],
        "decisions": [dataclass_to_dict(item) for item in decisions],
        "pending_order_reviews": [dataclass_to_dict(item) for item in pending_order_reviews],
        "held_position_signals": [dataclass_to_dict(item) for item in held_position_signals],
        "order_plans": [dataclass_to_dict(item) for item in order_plans],
        "execution_results": execution_results,
        "llm_usage": llm_usage,
        "run_metrics": run_metrics,
    }


def write_run_report(
    *,
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
) -> dict[str, Any]:
    payload = build_run_report_payload(
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
    write_json(run_path / "report.json", payload)
    (run_path / "report.html").write_text(_render_html(payload), encoding="utf-8")
    
    # Generate the specialized AI debug log
    write_ai_debug_log(
        run_path=run_path,
        selected_symbols=selected_symbols,
        debates=debates,
        decisions=decisions,
        execution_results=execution_results,
    )
    
    return payload


def write_ai_debug_log(
    *,
    run_path: Path,
    selected_symbols: list[SymbolMarketData],
    debates: list[SymbolDebate],
    decisions: list[TradeDecision],
    execution_results: list[dict],
) -> None:
    """Consolidates trades, reasons (debates), and market stats for AI debugging."""
    debug_entries = []
    
    # Map execution results for easy lookup
    exec_map = {res.get("symbol"): res for res in execution_results}
    # Map debates for easy lookup
    debate_map = {d.symbol: d for d in debates}
    # Map symbols for easy lookup
    symbol_map = {s.symbol: s for s in selected_symbols}

    for dec in decisions:
        symbol = dec.symbol
        debate = debate_map.get(symbol)
        mkt = symbol_map.get(symbol)
        exec_res = exec_map.get(symbol)

        entry = {
            "symbol": symbol,
            "decision": {
                "action": dec.action,
                "confidence": dec.confidence,
                "allocation": dec.allocation,
                "catalyst": dec.catalyst,
                "target_price": dec.target_price,
                "invalidation_price": dec.invalidation_price,
                "reward_risk_ratio": dec.reward_risk_ratio,
            },
            "reasoning": {
                "bull_case": {
                    "confidence": debate.bull_case.confidence if debate else None,
                    "arguments": debate.bull_case.arguments if debate else [],
                    "risks": debate.bull_case.risks if debate else [],
                },
                "bear_case": {
                    "confidence": debate.bear_case.confidence if debate else None,
                    "arguments": debate.bear_case.arguments if debate else [],
                    "risks": debate.bear_case.risks if debate else [],
                }
            },
            "market_data_snapshot": {
                "close": mkt.close if mkt else None,
                "indicators": dataclass_to_dict(mkt.indicators) if mkt else {},
                "premarket_gap_pct": mkt.premarket.gap_pct if mkt and mkt.premarket else None,
                "volume_ratio": mkt.raw_metrics.get("volume_ratio") if mkt else None,
            } if mkt else None,
            "execution": exec_res if exec_res else {"status": "no_order_placed"}
        }
        debug_entries.append(entry)

    write_json(run_path / "ai_debug_log.json", debug_entries)


def _render_html(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload["selected_symbols"]:
        flags = ", ".join(item.get("data_quality_flags", []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('symbol', '')))}</td>"
            f"<td>{float(item.get('score_breakdown', {}).get('total', 0.0)):.2f}</td>"
            f"<td>{html.escape(str(item.get('is_tradeable', True)))}</td>"
            f"<td>{html.escape(flags)}</td>"
            "</tr>"
        )
    order_rows = []
    for plan in payload["order_plans"]:
        order_rows.append(
            "<tr>"
            f"<td>{html.escape(str(plan.get('symbol', '')))}</td>"
            f"<td>{html.escape(str(plan.get('side', '')))}</td>"
            f"<td>{html.escape(str(plan.get('qty', '')))}</td>"
            f"<td>{html.escape(str(plan.get('entry_limit_price', '')))}</td>"
            f"<td>{html.escape(str(plan.get('stop_price', '')))}</td>"
            f"<td>{html.escape(str(plan.get('take_profit_price', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Trading Run Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Trading Run Report</h1>
  <p>Elapsed: <code>{html.escape(str(payload["summary"].get("elapsed_human", "")))}</code></p>
  <h2>Selected Symbols</h2>
  <table>
    <thead><tr><th>Symbol</th><th>Score</th><th>Tradeable</th><th>Quality Flags</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Order Plans</h2>
  <table>
    <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Stop</th><th>Target</th></tr></thead>
    <tbody>{''.join(order_rows)}</tbody>
  </table>
</body>
</html>
"""
