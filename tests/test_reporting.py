from trading_system.main import (
    build_mock_order_plans,
    load_mock_universe,
    mock_debates,
    mock_decisions,
)
from trading_system.reporting import write_run_report


def test_write_run_report_creates_json_and_html(tmp_path):
    selected = load_mock_universe()[:3]
    selected[0].data_quality_flags = ["extreme_20d_return"]
    selected[0].is_tradeable = False
    debates = mock_debates(selected)
    decisions = mock_decisions(debates)
    order_plans = build_mock_order_plans(decisions, selected)
    execution_results = [
        {"symbol": order_plans[0].symbol, "status": "dry_run", "qty": order_plans[0].qty, "side": order_plans[0].side}
    ]

    payload = write_run_report(
        run_path=tmp_path,
        selected_symbols=selected,
        debates=debates,
        decisions=decisions,
        pending_order_reviews=[],
        held_position_signals=[],
        order_plans=order_plans,
        execution_results=execution_results,
        llm_usage={"totals": {"call_count": 0, "total_tokens": 0}},
        run_metrics={"elapsed_human": "1.00s"},
    )

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
    assert payload["data_quality_warnings"][0]["symbol"] == selected[0].symbol
    assert "Trading Run Report" in (tmp_path / "report.html").read_text(encoding="utf-8")
