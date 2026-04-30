from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_system.utils import ensure_dir, read_json, write_json


def build_weekly_review(reports: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [trade for report in reports for trade in report.get("all_trades", [])]
    performance = _performance(trades)
    side_breakdown = _breakdown(trades, lambda trade: str(trade.get("side", "unknown")))
    exit_reason_breakdown = _breakdown(trades, lambda trade: str(trade.get("exit_reason", "unknown")))
    confidence_bucket_breakdown = _breakdown(trades, lambda trade: _confidence_bucket(trade))
    regime_breakdown = _regime_breakdown(trades)
    symbol_concentration = _symbol_concentration(trades, performance["net_pnl"])
    partial_exit_contribution = _performance(
        [trade for trade in trades if trade.get("exit_reason") == "partial_target_hit"]
    )
    conditional_hold_contribution = _performance(
        [trade for trade in trades if "hold" in str(trade.get("exit_reason", ""))]
    )
    tax_shadow_findings = _merge_dicts(report.get("tax_shadow_analysis", {}) for report in reports)
    exit_counterfactual_findings = _merge_dicts(
        report.get("exit_counterfactuals", {}) for report in reports
    )
    review = {
        "generated_at": datetime.now(UTC).isoformat(),
        "performance": performance,
        "top_winners": sorted(trades, key=lambda trade: trade.get("net_pnl", 0), reverse=True)[:5],
        "top_losers": sorted(trades, key=lambda trade: trade.get("net_pnl", 0))[:5],
        "symbol_concentration": symbol_concentration,
        "side_breakdown": side_breakdown,
        "exit_reason_breakdown": exit_reason_breakdown,
        "confidence_bucket_breakdown": confidence_bucket_breakdown,
        "regime_breakdown": regime_breakdown,
        "partial_exit_contribution": partial_exit_contribution,
        "conditional_hold_contribution": conditional_hold_contribution,
        "tax_shadow_findings": tax_shadow_findings,
        "exit_counterfactual_findings": exit_counterfactual_findings,
    }
    review["adaptation_candidates"] = _adaptation_candidates(review)
    return review


def write_weekly_review(
    reports: list[dict[str, Any]],
    *,
    output_dir: Path,
    stem: str = "weekly_trade_review",
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    review = build_weekly_review(reports)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    write_json(json_path, review)
    markdown_path.write_text(render_weekly_markdown(review), encoding="utf-8")
    return json_path, markdown_path


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    return [read_json(path) for path in paths]


def render_weekly_markdown(review: dict[str, Any]) -> str:
    performance = review["performance"]
    lines = [
        "# Weekly Trade Review",
        "",
        "## Performance",
        f"- Trades: {performance['total_trades']}",
        f"- Net P/L: {performance['net_pnl']:.2f}",
        f"- Win rate: {performance['win_rate']:.2%}",
        f"- Profit factor: {performance['profit_factor']}",
        "",
        "## Adaptation Candidates",
    ]
    candidates = review.get("adaptation_candidates", [])
    if not candidates:
        lines.append("- None")
    for candidate in candidates:
        lines.append(
            f"- {candidate['type']}: {candidate['evidence']} Suggested experiment: {candidate['suggested_experiment']}"
        )
    return "\n".join(lines) + "\n"


def _performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades]
    returns = [float(trade.get("return_pct", 0.0) or 0.0) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_wins = sum(wins)
    gross_losses = -sum(losses)
    return {
        "total_trades": len(trades),
        "net_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "profit_factor": round(gross_wins / gross_losses, 4) if gross_losses else None if gross_wins else 0.0,
        "average_return_pct": round(sum(returns) / len(returns), 4) if returns else 0.0,
        "median_return_pct": round(statistics.median(returns), 4) if returns else 0.0,
        "average_winner": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "average_loser": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "gross_wins": round(gross_wins, 2),
        "gross_losses": round(gross_losses, 2),
    }


def _breakdown(trades: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(key_fn(trade), []).append(trade)
    return {key: _performance(items) for key, items in sorted(grouped.items())}


def _regime_breakdown(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        regime = trade.get("regime_snapshot") or {}
        if not regime:
            grouped.setdefault("unknown", []).append(trade)
            continue
        for key, value in regime.items():
            grouped.setdefault(f"{key}={value}", []).append(trade)
    return {key: _performance(items) for key, items in sorted(grouped.items())}


def _confidence_bucket(trade: dict[str, Any]) -> str:
    confidence = float(trade.get("confidence", 0.0) or 0.0)
    lower = min(0.9, max(0.0, int(confidence * 10) / 10))
    upper = 1.0 if lower >= 0.9 else lower + 0.1
    return f"{lower:.2f}-{upper:.2f}"


def _symbol_concentration(trades: list[dict[str, Any]], total_net_pnl: float) -> dict[str, Any]:
    by_symbol = _breakdown(trades, lambda trade: str(trade.get("symbol", "unknown")))
    if not by_symbol:
        return {"top_symbol": None, "top_symbol_net_pnl": 0.0, "top_symbol_profit_share": 0.0}
    top_symbol, top = max(by_symbol.items(), key=lambda item: item[1]["net_pnl"])
    share = round(top["net_pnl"] / total_net_pnl, 4) if total_net_pnl else 0.0
    return {
        "by_symbol": by_symbol,
        "top_symbol": top_symbol,
        "top_symbol_net_pnl": top["net_pnl"],
        "top_symbol_profit_share": share,
    }


def _merge_dicts(values) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _adaptation_candidates(review: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for regime, stats in review["regime_breakdown"].items():
        if stats["total_trades"] and (stats["net_pnl"] < 0 or (stats["profit_factor"] is not None and stats["profit_factor"] < 1)):
            candidates.append(
                {
                    "type": "weak_regime",
                    "evidence": f"{regime} net_pnl={stats['net_pnl']} profit_factor={stats['profit_factor']}",
                    "suggested_experiment": "A/B test reduced sizing or stricter entry filters in this regime.",
                }
            )
    for reason, stats in review["exit_reason_breakdown"].items():
        if stats["total_trades"] and stats["net_pnl"] < 0:
            candidates.append(
                {
                    "type": "weak_exit_reason",
                    "evidence": f"{reason} net_pnl={stats['net_pnl']}",
                    "suggested_experiment": "Replay exits with alternative stop/hold rules before changing live parameters.",
                }
            )
    buckets = review["confidence_bucket_breakdown"]
    high = buckets.get("0.90-1.00")
    lower_positive = [
        stats for bucket, stats in buckets.items() if bucket != "0.90-1.00" and stats["net_pnl"] > 0
    ]
    if high and lower_positive and high["net_pnl"] < max(stats["net_pnl"] for stats in lower_positive):
        candidates.append(
            {
                "type": "confidence_inversion",
                "evidence": f"0.90-1.00 bucket net_pnl={high['net_pnl']} underperformed a lower bucket.",
                "suggested_experiment": "Review calibration and confidence caps before increasing high-confidence size.",
            }
        )
    concentration = review["symbol_concentration"]
    if abs(concentration.get("top_symbol_profit_share", 0.0)) > 0.35:
        candidates.append(
            {
                "type": "profit_concentration",
                "evidence": (
                    f"{concentration['top_symbol']} contributed "
                    f"{concentration['top_symbol_profit_share']:.2%} of net P/L."
                ),
                "suggested_experiment": "Run acceptance without the top symbol and compare robustness.",
            }
        )
    return candidates
