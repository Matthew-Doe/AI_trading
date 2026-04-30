from __future__ import annotations

from trading_system.weekly_review import build_weekly_review, render_weekly_markdown, write_weekly_review


def _report() -> dict:
    return {
        "all_trades": [
            {
                "symbol": "AAPL",
                "side": "long",
                "net_pnl": 120.0,
                "return_pct": 0.12,
                "confidence": 0.92,
                "exit_reason": "target_hit",
                "qty": 10,
                "regime_snapshot": {"trend_regime": "uptrend"},
            },
            {
                "symbol": "TSLA",
                "side": "long",
                "net_pnl": -100.0,
                "return_pct": -0.10,
                "confidence": 0.95,
                "exit_reason": "stop_loss",
                "qty": 5,
                "regime_snapshot": {"trend_regime": "downtrend"},
            },
            {
                "symbol": "AAPL",
                "side": "long",
                "net_pnl": 80.0,
                "return_pct": 0.08,
                "confidence": 0.65,
                "exit_reason": "partial_target_hit",
                "qty": 4,
                "regime_snapshot": {"trend_regime": "uptrend"},
            },
        ],
        "tax_shadow_analysis": {"blocked_trade_count": 1, "estimated_net_pnl": 40.0},
        "exit_counterfactuals": {"summary": {"thesis_failed": {"average_delta_vs_actual": 12.0}}},
    }


def test_weekly_review_summarizes_performance_breakdowns_and_contributions():
    review = build_weekly_review([_report()])

    assert review["performance"]["total_trades"] == 3
    assert review["performance"]["net_pnl"] == 100.0
    assert review["performance"]["win_rate"] == 0.6667
    assert review["performance"]["profit_factor"] == 2.0
    assert review["symbol_concentration"]["top_symbol"] == "AAPL"
    assert review["symbol_concentration"]["top_symbol_profit_share"] == 2.0
    assert review["exit_reason_breakdown"]["stop_loss"]["net_pnl"] == -100.0
    assert review["confidence_bucket_breakdown"]["0.90-1.00"]["net_pnl"] == 20.0
    assert review["regime_breakdown"]["trend_regime=downtrend"]["net_pnl"] == -100.0
    assert review["partial_exit_contribution"]["net_pnl"] == 80.0
    assert review["tax_shadow_findings"]["blocked_trade_count"] == 1


def test_weekly_review_flags_adaptation_candidates_and_renders_markdown():
    review = build_weekly_review([_report()])

    candidate_types = {item["type"] for item in review["adaptation_candidates"]}
    assert "weak_regime" in candidate_types
    assert "weak_exit_reason" in candidate_types
    assert "confidence_inversion" in candidate_types
    assert "profit_concentration" in candidate_types

    markdown = render_weekly_markdown(review)
    assert "Adaptation Candidates" in markdown
    assert "weak_regime" in markdown


def test_write_weekly_review_outputs_json_and_markdown(tmp_path):
    json_path, markdown_path = write_weekly_review([_report()], output_dir=tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert "Weekly Trade Review" in markdown_path.read_text(encoding="utf-8")
