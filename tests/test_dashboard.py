import json
from types import SimpleNamespace

import pandas as pd

from trading_system.dashboard import (
    DashboardRequestHandler,
    build_dashboard_payload,
    build_performance_payload_from_history,
    benchmark_history_from_frame,
    find_latest_run_id,
    find_latest_log_run_id,
)
from trading_system.main import (
    build_mock_order_plans,
    build_run_metrics,
    load_mock_universe,
    mock_debates,
    mock_decisions,
)
from trading_system.reporting import write_run_report
from trading_system.utils import write_json


def _write_mock_run(tmp_path, run_id="20260424T123001Z"):
    run_path = tmp_path / "runs" / run_id
    run_path.mkdir(parents=True)
    selected = load_mock_universe()[:3]
    selected[0].data_quality_flags = ["extreme_20d_return"]
    selected[0].is_tradeable = False
    debates = mock_debates(selected)
    decisions = mock_decisions(debates)
    order_plans = build_mock_order_plans(decisions, selected)
    execution_results = [
        {
            "symbol": order_plans[0].symbol,
            "status": "dry_run",
            "qty": order_plans[0].qty,
            "side": order_plans[0].side,
        }
    ]
    run_metrics = build_run_metrics(
        selected[0].premarket.timestamp_dt,
        selected[0].premarket.timestamp_dt,
        1.0,
    ) if hasattr(selected[0].premarket, "timestamp_dt") else {"elapsed_human": "1.00s"}
    write_json(run_path / "selected_symbols.json", selected)
    write_json(run_path / "debates.json", debates)
    write_json(run_path / "decisions.json", decisions)
    write_json(run_path / "pending_order_reviews.json", [])
    write_json(run_path / "held_position_signals.json", [])
    write_json(run_path / "order_plans.json", order_plans)
    write_json(run_path / "execution_results.json", execution_results)
    write_json(run_path / "llm_usage.json", {"totals": {"call_count": 0, "total_tokens": 0}})
    write_json(run_path / "run_metrics.json", run_metrics)
    write_run_report(
        run_path=run_path,
        selected_symbols=selected,
        debates=debates,
        decisions=decisions,
        pending_order_reviews=[],
        held_position_signals=[],
        order_plans=order_plans,
        execution_results=execution_results,
        llm_usage={"totals": {"call_count": 0, "total_tokens": 0}},
        run_metrics=run_metrics,
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / f"{run_id}.log").write_text("line 1\nline 2\n", encoding="utf-8")
    return run_id


def test_find_latest_run_id_ignores_incomplete_runs(tmp_path):
    complete = _write_mock_run(tmp_path, "20260424T123001Z")
    incomplete = tmp_path / "runs" / "20260425T123001Z"
    incomplete.mkdir()

    assert find_latest_run_id(tmp_path / "runs") == complete


def test_build_dashboard_payload_includes_latest_run_and_log_tail(tmp_path):
    run_id = _write_mock_run(tmp_path)

    payload = build_dashboard_payload(tmp_path / "runs", tmp_path / "logs")

    assert payload["latest_run"]["run_id"] == run_id
    assert payload["latest_run"]["summary"]["selected_symbols"] == 3
    assert payload["latest_run"]["data_quality_warnings"][0]["symbol"]
    assert payload["latest_log_tail"] == ["line 1", "line 2"]


def test_build_dashboard_payload_uses_newest_log_for_in_progress_run(tmp_path):
    complete_run_id = _write_mock_run(tmp_path, "20260424T123001Z")
    active_log = tmp_path / "logs" / "20260425T123001Z.log"
    active_log.write_text("starting\nbuilding universe\n", encoding="utf-8")

    payload = build_dashboard_payload(tmp_path / "runs", tmp_path / "logs")

    assert payload["latest_run"]["run_id"] == complete_run_id
    assert payload["active_run_id"] == "20260425T123001Z"
    assert payload["latest_log_tail"] == ["starting", "building universe"]
    assert find_latest_log_run_id(tmp_path / "logs") == "20260425T123001Z"


def test_dashboard_handler_serves_latest_api_json(tmp_path):
    run_id = _write_mock_run(tmp_path)

    handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
    handler.server = SimpleNamespace(run_dir=tmp_path / "runs", log_dir=tmp_path / "logs")
    payload = handler._route_json("/api/latest")

    assert payload["latest_run"]["run_id"] == run_id


def test_dashboard_handler_rejects_unknown_api_path(tmp_path):
    _write_mock_run(tmp_path)
    handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
    handler.server = SimpleNamespace(run_dir=tmp_path / "runs", log_dir=tmp_path / "logs")

    try:
        handler._route_json("/api/missing")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("unknown API path should raise FileNotFoundError")


def test_build_performance_payload_normalizes_portfolio_against_benchmark():
    portfolio_history = {
        "timestamp": [1713744000, 1713830400, 1713916800],
        "equity": [100000.0, 101000.0, 99000.0],
    }
    benchmark_history = [
        {"date": "2024-04-22", "close": 500.0},
        {"date": "2024-04-23", "close": 505.0},
        {"date": "2024-04-24", "close": 495.0},
    ]

    payload = build_performance_payload_from_history(
        portfolio_history,
        benchmark_history,
        benchmark_symbol="SPY",
    )

    assert payload["benchmark_symbol"] == "SPY"
    assert payload["points"][0]["portfolio_return_pct"] == 0.0
    assert payload["points"][1]["portfolio_return_pct"] == 1.0
    assert payload["points"][2]["portfolio_return_pct"] == -1.0
    assert payload["points"][1]["benchmark_return_pct"] == 1.0
    assert payload["points"][2]["benchmark_return_pct"] == -1.0


def test_dashboard_handler_serves_performance_api(tmp_path):
    _write_mock_run(tmp_path)
    handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
    handler.server = SimpleNamespace(
        run_dir=tmp_path / "runs",
        log_dir=tmp_path / "logs",
        performance_payload={"points": [], "benchmark_symbol": "SPY", "error": "offline"},
    )

    payload = handler._route_json("/api/performance")

    assert payload["benchmark_symbol"] == "SPY"
    assert payload["error"] == "offline"


def test_benchmark_history_from_frame_handles_yfinance_multiindex_columns():
    frame = pd.DataFrame(
        {
            ("Close", "SPY"): [500.0, 505.0],
            ("Open", "SPY"): [499.0, 501.0],
        },
        index=pd.to_datetime(["2024-04-22", "2024-04-23"]),
    )

    history = benchmark_history_from_frame(frame)

    assert history == [
        {"date": "2024-04-22", "close": 500.0},
        {"date": "2024-04-23", "close": 505.0},
    ]
