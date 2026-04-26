from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
import yfinance as yf

from trading_system.config import TradingConfig
from trading_system.utils import read_json


REQUIRED_RUN_FILES = (
    "selected_symbols.json",
    "decisions.json",
    "order_plans.json",
    "execution_results.json",
    "run_metrics.json",
)


def get_period_params(period: str) -> tuple[str | None, str, str | None]:
    """Returns (alpaca_period, timeframe, start_timestamp)"""
    if period == "day":
        return "1D", "5Min", None
    if period == "week":
        return "1W", "5Min", None
    if period == "month":
        return "1M", "1D", None
    if period == "ytd":
        # Jan 1st 2026
        return None, "1D", "2026-01-01T00:00:00Z"
    if period == "1yr":
        return "1A", "1D", None
    if period == "max":
        return None, "1D", "2026-04-14T00:00:00Z"
    return None, "1D", "2026-04-14T00:00:00Z"


def find_latest_run_id(run_dir: Path) -> str | None:
    if not run_dir.exists():
        return None
    for path in sorted((item for item in run_dir.iterdir() if item.is_dir()), reverse=True):
        if all((path / name).exists() for name in REQUIRED_RUN_FILES):
            return path.name
    return None


def list_run_ids(run_dir: Path, limit: int = 30) -> list[str]:
    if not run_dir.exists():
        return []
    run_ids = [
        path.name
        for path in sorted((item for item in run_dir.iterdir() if item.is_dir()), reverse=True)
        if all((path / name).exists() for name in REQUIRED_RUN_FILES)
    ]
    return run_ids[:limit]


def find_latest_log_run_id(log_dir: Path) -> str | None:
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in logs:
        if path.name == "market_close_summary.log":
            continue
        return path.stem
    return None


def build_dashboard_payload(run_dir: Path, log_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    selected_run_id = run_id or find_latest_run_id(run_dir)
    active_run_id = find_latest_log_run_id(log_dir)
    log_run_id = run_id or active_run_id or selected_run_id
    return {
        "runs": list_run_ids(run_dir),
        "active_run_id": active_run_id,
        "latest_run": load_run_payload(run_dir, selected_run_id) if selected_run_id else None,
        "latest_log_tail": read_log_tail(log_dir, log_run_id),
    }


def load_run_payload(run_dir: Path, run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    run_path = run_dir / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    report_path = run_path / "report.json"
    if report_path.exists():
        payload = read_json(report_path)
    else:
        payload = {
            "summary": _build_summary_from_artifacts(run_path),
            "data_quality_warnings": _data_quality_warnings(run_path),
            "selected_symbols": _read_optional_json(run_path / "selected_symbols.json", []),
            "decisions": _read_optional_json(run_path / "decisions.json", []),
            "pending_order_reviews": _read_optional_json(run_path / "pending_order_reviews.json", []),
            "held_position_signals": _read_optional_json(run_path / "held_position_signals.json", []),
            "order_plans": _read_optional_json(run_path / "order_plans.json", []),
            "execution_results": _read_optional_json(run_path / "execution_results.json", []),
            "llm_usage": _read_optional_json(run_path / "llm_usage.json", {}),
            "run_metrics": _read_optional_json(run_path / "run_metrics.json", {}),
        }
    payload["run_id"] = run_id
    payload["artifact_links"] = {
        "summary": f"/artifacts/{run_id}/summary.txt",
        "report": f"/artifacts/{run_id}/report.html",
    }
    return payload


def read_log_tail(log_dir: Path, run_id: str | None, lines: int = 120) -> list[str]:
    if not run_id:
        return []
    path = log_dir / f"{run_id}.log"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def fetch_performance_payload(config: TradingConfig, period: str = "max") -> dict[str, Any]:
    try:
        alpaca_period, timeframe, start = get_period_params(period)
        portfolio_history = fetch_alpaca_portfolio_history(
            config, period=alpaca_period, timeframe=timeframe, start=start
        )
        start_date, end_date = portfolio_history_date_bounds(portfolio_history)
        benchmark_history = fetch_benchmark_history(
            config.benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            config=config,
        )
        return build_performance_payload_from_history(
            portfolio_history,
            benchmark_history,
            benchmark_symbol=config.benchmark_symbol,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "benchmark_symbol": config.benchmark_symbol,
            "points": [],
            "error": str(exc),
        }


def fetch_alpaca_portfolio_history(
    config: TradingConfig,
    *,
    period: str | None = None,
    timeframe: str = "1D",
    start: str | None = None,
) -> dict[str, Any]:
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        raise ValueError("Alpaca credentials are not configured.")

    params: dict[str, Any] = {
        "timeframe": timeframe,
        "extended_hours": "false",
    }

    if start:
        params["start"] = start
    elif period:
        params["period"] = period
    else:
        # Default to inception
        params["start"] = "2026-04-14T00:00:00Z"

    response = requests.get(
        f"{config.alpaca_paper_base_url.rstrip('/')}/v2/account/portfolio/history",
        headers={
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_secret_key,
        },
        params=params,
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def portfolio_history_date_bounds(portfolio_history: dict[str, Any]) -> tuple[str | None, str | None]:
    timestamps = [int(item) for item in portfolio_history.get("timestamp", []) if item is not None]
    if not timestamps:
        return None, None
    start = datetime.fromtimestamp(min(timestamps), UTC).date().isoformat()
    end = datetime.fromtimestamp(max(timestamps), UTC).date().isoformat()
    return start, end


def fetch_benchmark_history(
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: str = "1Day",
    config: TradingConfig | None = None,
) -> list[dict[str, Any]]:
    if config and config.alpaca_api_key and config.alpaca_secret_key:
        try:
            history = fetch_benchmark_history_alpaca(
                symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                config=config,
            )
            if history:
                return history
        except Exception:
            pass
    
    # Fallback to yfinance
    yf_interval = "1d"
    if timeframe == "1Min":
        yf_interval = "1m"
    elif timeframe == "5Min":
        yf_interval = "5m"
    elif timeframe == "1Hour":
        yf_interval = "1h"

    frame = yf.download(
        tickers=symbol,
        period="max" if start_date is None else None,
        start=start_date,
        end=None if timeframe in ("1Min", "5Min", "1Hour") else end_date,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No benchmark history returned for {symbol}.")
    return benchmark_history_from_frame(frame)


def fetch_benchmark_history_alpaca(
    symbol: str,
    *,
    start_date: str | None,
    end_date: str | None,
    timeframe: str = "1Day",
    config: TradingConfig,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
        "sort": "asc",
    }
    if start_date:
        params["start"] = f"{start_date}T00:00:00Z"
    if end_date:
        params["end"] = f"{end_date}T23:59:59Z"
    
    # For intraday, don't set end_date to ensure we get latest
    if timeframe != "1Day":
        params.pop("end", None)

    response = requests.get(
        "https://data.alpaca.markets/v2/stocks/bars",
        headers={
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_secret_key,
        },
        params=params,
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    bars = response.json().get("bars", {}).get(symbol, [])
    return [
        {
            "date": bar["t"],  # Keep full timestamp for join
            "close": float(bar["c"]),
        }
        for bar in bars
        if bar.get("t") and bar.get("c") is not None
    ]


def benchmark_history_from_frame(frame) -> list[dict[str, Any]]:
    close_data = frame["Close"]
    if hasattr(close_data, "columns"):
        close_data = close_data.iloc[:, 0]
    closes = close_data.dropna()
    return [
        {
            "date": index.date().isoformat() if hasattr(index, "date") else str(index),
            "close": float(close),
        }
        for index, close in closes.items()
    ]


def build_performance_payload_from_history(
    portfolio_history: dict[str, Any],
    benchmark_history: list[dict[str, Any]],
    *,
    benchmark_symbol: str,
) -> dict[str, Any]:
    timestamps = portfolio_history.get("timestamp", [])
    equities = portfolio_history.get("equity", [])
    
    # Pre-process benchmark for faster lookup
    # Use floor-normalized timestamps (minute-level) for better joining
    benchmark_lookup = {}
    for item in benchmark_history:
        b_date = item["date"]
        # Standardize ISO string to YYYY-MM-DDTHH:MM
        if "T" in b_date:
            try:
                dt = datetime.fromisoformat(b_date.replace("Z", "+00:00"))
                norm_key = dt.strftime("%Y-%m-%dT%H:%M")
                benchmark_lookup[norm_key] = float(item["close"])
            except ValueError:
                pass
        
        # Always store the date-only version too
        date_only = b_date.split("T")[0]
        if date_only not in benchmark_lookup:
            benchmark_lookup[date_only] = float(item["close"])

    joined: list[dict[str, Any]] = []
    for timestamp, equity in zip(timestamps, equities, strict=False):
        if equity is None:
            continue
        dt = datetime.fromtimestamp(int(timestamp), UTC)
        
        norm_key = dt.strftime("%Y-%m-%dT%H:%M")
        date_key = dt.date().isoformat()
        
        # Try minute-level match, then date match
        benchmark_close = benchmark_lookup.get(norm_key) or benchmark_lookup.get(date_key)

        if benchmark_close is None:
            continue
            
        joined.append(
            {
                "date": dt.isoformat(),
                "portfolio_value": float(equity),
                "benchmark_close": benchmark_close,
            }
        )

    if not joined:
        return {
            "benchmark_symbol": benchmark_symbol,
            "points": [],
            "error": "No overlapping portfolio and benchmark history.",
        }

    # Alpaca's base_value is the equity at the start of the period
    base_portfolio = float(portfolio_history.get("base_value", joined[0]["portfolio_value"]))
    if base_portfolio <= 0:
        base_portfolio = joined[0]["portfolio_value"]
        
    base_benchmark = joined[0]["benchmark_close"]
    
    if base_portfolio <= 0 or base_benchmark <= 0:
        return {
            "benchmark_symbol": benchmark_symbol,
            "points": [],
            "error": f"Invalid baseline: P={base_portfolio}, B={base_benchmark}",
        }

    points = []
    negative_days = 0
    underperform_days = 0
    
    # To count days accurately, we track seen dates
    seen_dates = set()

    for item in joined:
        p_ret = ((item["portfolio_value"] / base_portfolio) - 1.0) * 100
        b_ret = ((item["benchmark_close"] / base_benchmark) - 1.0) * 100
        
        # Determine if this "point" represents a unique day for daily stats
        # If joined data is daily, this is simple. If intraday, we only count the close/last point of each day.
        # But for 'Max' it is daily.
        date_str = item["date"].split("T")[0]
        
        points.append(
            {
                "date": item["date"][:16].replace("T", " "),
                "portfolio_value": round(item["portfolio_value"], 2),
                "benchmark_close": round(item["benchmark_close"], 4),
                "portfolio_return_pct": round(p_ret, 4),
                "benchmark_return_pct": round(b_ret, 4),
            }
        )
    
    # Calculate daily stats from the points
    # We look at the LAST point of each unique day to determine if that day was negative or underperforming
    daily_summaries = {}
    for pt in points:
        d = pt["date"].split(" ")[0]
        daily_summaries[d] = pt

    for d in daily_summaries:
        pt = daily_summaries[d]
        if pt["portfolio_return_pct"] < 0:
            negative_days += 1
        if pt["portfolio_return_pct"] < pt["benchmark_return_pct"]:
            underperform_days += 1

    return {
        "benchmark_symbol": benchmark_symbol,
        "points": points,
        "negative_days": negative_days,
        "underperform_days": underperform_days,
        "total_days": len(daily_summaries),
        "error": None,
    }


def _read_optional_json(path: Path, default: Any) -> Any:
    return read_json(path) if path.exists() else default


def _build_summary_from_artifacts(run_path: Path) -> dict[str, Any]:
    selected = _read_optional_json(run_path / "selected_symbols.json", [])
    decisions = _read_optional_json(run_path / "decisions.json", [])
    orders = _read_optional_json(run_path / "order_plans.json", [])
    execution = _read_optional_json(run_path / "execution_results.json", [])
    metrics = _read_optional_json(run_path / "run_metrics.json", {})
    return {
        "selected_symbols": len(selected),
        "active_decisions": len([item for item in decisions if item.get("action") != "skip"]),
        "planned_orders": len(orders),
        "execution_results": len(execution),
        "elapsed_human": metrics.get("elapsed_human"),
    }


def _data_quality_warnings(run_path: Path) -> list[dict[str, Any]]:
    selected = _read_optional_json(run_path / "selected_symbols.json", [])
    return [
        {
            "symbol": item.get("symbol"),
            "flags": item.get("data_quality_flags", []),
            "is_tradeable": item.get("is_tradeable", True),
        }
        for item in selected
        if item.get("data_quality_flags") or item.get("is_tradeable") is False
    ]


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        handler_class,
        *,
        run_dir: Path,
        log_dir: Path,
        config: TradingConfig | None = None,
    ):
        super().__init__(server_address, handler_class)
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.config = config or TradingConfig()
        self._perf_cache_multi: dict[str, Any] = {}
        self._perf_cache_saved_at_multi: dict[str, float] = {}

    def performance_payload(self, period: str = "max") -> dict[str, Any]:
        now = time.monotonic()
        if period in self._perf_cache_multi and now - self._perf_cache_saved_at_multi.get(period, 0) < 300:
            return self._perf_cache_multi[period]
        self._perf_cache_multi[period] = fetch_performance_payload(self.config, period=period)
        self._perf_cache_saved_at_multi[period] = now
        return self._perf_cache_multi[period]


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/" or parsed.path == "/index.html":
                self._send_text(HTTPStatus.OK, render_dashboard_html(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/api/"):
                self._send_json(HTTPStatus.OK, self._route_json(parsed))
                return
            if parsed.path.startswith("/artifacts/"):
                self._send_artifact(parsed.path)
                return
            raise FileNotFoundError(parsed.path)
        except FileNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route_json(self, parsed: Any) -> dict[str, Any] | list[str]:
        if isinstance(parsed, str):
            parsed = urlparse(parsed)
        path = parsed.path
        if path == "/api/latest":
            return build_dashboard_payload(self.server.run_dir, self.server.log_dir)
        if path == "/api/runs":
            return list_run_ids(self.server.run_dir)
        if path == "/api/logs/latest":
            run_id = find_latest_run_id(self.server.run_dir)
            return {"run_id": run_id, "lines": read_log_tail(self.server.log_dir, run_id)}
        if path == "/api/performance":
            params = parse_qs(parsed.query)
            period = params.get("period", ["max"])[0]
            if hasattr(self.server, "performance_payload") and callable(self.server.performance_payload):
                return self.server.performance_payload(period=period)
            if isinstance(getattr(self.server, "performance_payload", None), dict):
                return self.server.performance_payload
            return {"points": [], "benchmark_symbol": "SPY", "error": "performance unavailable"}
        prefix = "/api/runs/"
        if path.startswith(prefix):
            run_id = unquote(path.removeprefix(prefix)).strip("/")
            payload = build_dashboard_payload(self.server.run_dir, self.server.log_dir, run_id)
            if payload["latest_run"] is None:
                raise FileNotFoundError(path)
            return payload
        raise FileNotFoundError(path)

    def _send_artifact(self, path: str) -> None:
        parts = path.strip("/").split("/", maxsplit=2)
        if len(parts) != 3:
            raise FileNotFoundError(path)
        _, run_id, filename = parts
        if "/" in filename or filename not in {"summary.txt", "report.html", "report.json"}:
            raise FileNotFoundError(path)
        artifact = self.server.run_dir / run_id / filename
        if not artifact.exists():
            raise FileNotFoundError(path)
        content_type = "text/html; charset=utf-8" if filename.endswith(".html") else "text/plain; charset=utf-8"
        if filename.endswith(".json"):
            content_type = "application/json; charset=utf-8"
        self._send_text(HTTPStatus.OK, artifact.read_text(encoding="utf-8"), content_type)

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        self._send_text(
            status,
            json.dumps(payload, indent=2, sort_keys=True),
            "application/json; charset=utf-8",
        )

    def _send_text(self, status: HTTPStatus, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def render_dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Trading Dashboard</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #f5f7fa; color: #1f2933; }
    header { background: #102a43; color: white; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; }
    main { padding: 18px 20px 28px; display: grid; gap: 16px; }
    section { background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; }
    h1 { font-size: 20px; margin: 0; }
    h2 { font-size: 15px; margin: 0 0 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4e7eb; padding: 7px 6px; text-align: left; vertical-align: top; }
    th { color: #52606d; font-weight: 600; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .metric { background: #f0f4f8; border-radius: 6px; padding: 10px; }
    .metric strong { display: block; font-size: 20px; margin-top: 4px; }
    .muted { color: #627d98; }
    .bad { color: #b42318; font-weight: 600; }
    pre { background: #102a43; color: #d9e2ec; border-radius: 6px; padding: 12px; overflow: auto; max-height: 360px; }
    a { color: #0b63ce; }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    
    .perf-container { position: relative; }
    .timerange-picker { margin-bottom: 12px; display: flex; gap: 8px; }
    .timerange-picker button { 
      background: #f0f4f8; border: 1px solid #d9e2ec; border-radius: 4px; 
      padding: 4px 12px; font-size: 12px; cursor: pointer; color: #52606d; font-weight: 600;
    }
    .timerange-picker button.active { background: #102a43; color: white; border-color: #102a43; }
    .timerange-picker button:hover:not(.active) { background: #e4e7eb; }
  </style>
</head>
<body>
  <header>
    <h1>AI Trading Dashboard</h1>
    <div id="status" class="muted">Loading...</div>
  </header>
  <main>
    <section><h2>Run</h2><div id="metrics" class="grid"></div></section>
    <section><h2>Portfolio Stats (Since Inception)</h2><div id="portfolio-stats" class="grid"></div></section>
    <section><h2>Portfolio Performance (Since Inception)</h2><div id="performance-inception" class="perf-container"></div></section>
    <section>
      <h2>Relative Performance</h2>
      <div class="timerange-picker">
        <button id="btn-day" onclick="changePeriod('day')">Day</button>
        <button id="btn-week" onclick="changePeriod('week')">Week</button>
        <button id="btn-month" onclick="changePeriod('month')">Month</button>
        <button id="btn-ytd" onclick="changePeriod('ytd')">YTD</button>
        <button id="btn-1yr" onclick="changePeriod('1yr')">1Y</button>
        <button id="btn-max" onclick="changePeriod('max')" class="active">Max</button>
      </div>
      <div id="performance-timerange" class="perf-container"></div>
    </section>
    <section><h2>Selected Symbols</h2><div id="selected"></div></section>
    <section><h2>Decisions</h2><div id="decisions"></div></section>
    <section><h2>Order Plans</h2><div id="orders"></div></section>
    <section><h2>Execution Results</h2><div id="execution"></div></section>
    <section><h2>Log Tail</h2><pre id="logs"></pre></section>
  </main>
  <script>
    const fmt = value => value === null || value === undefined ? "" : String(value);
    const charts = {};

    function table(columns, rows) {
      if (!rows || rows.length === 0) return '<p class="muted">No data.</p>';
      return '<table><thead><tr>' + columns.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map(row => '<tr>' + columns.map(c => `<td>${fmt(c[1](row))}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>';
    }

    function renderPerformance(payload, chartId) {
      if (!payload || payload.error) return `<p class="muted">${fmt(payload && payload.error ? payload.error : 'No performance data.')}</p>`;
      const points = payload.points || [];
      if (points.length === 0) return '<p class="muted">No performance data.</p>';
      const width = 900, height = 260, pad = 34;
      const values = points.flatMap(p => [p.portfolio_return_pct, p.benchmark_return_pct]);
      const min = Math.min(...values, 0), max = Math.max(...values, 0);
      const span = max - min || 1;
      const x = i => pad + (i / Math.max(points.length - 1, 1)) * (width - pad * 2);
      const y = v => height - pad - ((v - min) / span) * (height - pad * 2);
      const path = key => points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(' ');
      const last = points[points.length - 1];

      charts[chartId] = { points, width, height, pad, min, max, span };

      return `
        <svg id="svg-${chartId}" viewBox="0 0 ${width} ${height}" width="100%" height="260" style="overflow: visible; display: block;" onmousemove="showPerfTooltip(event, '${chartId}')" onmouseleave="hidePerfTooltip('${chartId}')">
          <g stroke="#bcccdc" stroke-dasharray="4">
            <line x1="${pad}" y1="${y(0)}" x2="${width - pad}" y2="${y(0)}" />
          </g>
          <path d="${path('portfolio_return_pct')}" fill="none" stroke="#0b63ce" stroke-width="3" stroke-linejoin="round" />
          <path d="${path('benchmark_return_pct')}" fill="none" stroke="#f59e0b" stroke-width="3" stroke-linejoin="round" />
          
          <g id="hover-${chartId}" style="display: none" pointer-events="none">
            <line id="line-${chartId}" x1="0" y1="${pad}" x2="0" y2="${height - pad}" stroke="#627d98" stroke-width="1" stroke-dasharray="2" />
            <circle id="dot-p-${chartId}" r="5" fill="#0b63ce" stroke="white" stroke-width="2" />
            <circle id="dot-b-${chartId}" r="5" fill="#f59e0b" stroke="white" stroke-width="2" />
            <g id="tooltip-${chartId}">
              <rect rx="4" fill="rgba(16, 42, 67, 0.95)" width="150" height="64" />
              <text id="txt-date-${chartId}" x="10" y="20" fill="white" font-size="12" font-weight="bold"></text>
              <text id="txt-p-${chartId}" x="10" y="38" fill="#91befa" font-size="12"></text>
              <text id="txt-b-${chartId}" x="10" y="54" fill="#ffca6b" font-size="12"></text>
            </g>
          </g>

          <text x="${pad}" y="18" fill="#0b63ce" font-weight="bold">Portfolio ${last.portfolio_return_pct.toFixed(2)}%</text>
          <text x="${pad + 160}" y="18" fill="#b45309" font-weight="bold">${payload.benchmark_symbol} ${last.benchmark_return_pct.toFixed(2)}%</text>
          <text x="${pad}" y="${height - 8}" fill="#627d98" font-size="12">${points[0].date}</text>
          <text x="${width - pad}" y="${height - 8}" fill="#627d98" font-size="12" text-anchor="end">${last.date}</text>
          
          <rect x="${pad}" y="${pad}" width="${width - pad * 2}" height="${height - pad * 2}" fill="transparent" pointer-events="all" />
        </svg>`;
    }

    function showPerfTooltip(e, chartId) {
      const data = charts[chartId];
      if (!data) return;
      
      const svg = document.getElementById(`svg-${chartId}`);
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const localPt = pt.matrixTransform(svg.getScreenCTM().inverse());
      const viewBoxX = localPt.x;
      
      let i = Math.round(((viewBoxX - data.pad) / (data.width - data.pad * 2)) * (data.points.length - 1));
      i = Math.max(0, Math.min(data.points.length - 1, i));
      
      const p = data.points[i];
      const hover = document.getElementById(`hover-${chartId}`);
      const line = document.getElementById(`line-${chartId}`);
      const dotP = document.getElementById(`dot-p-${chartId}`);
      const dotB = document.getElementById(`dot-b-${chartId}`);
      const tooltip = document.getElementById(`tooltip-${chartId}`);
      
      const x = data.pad + (i / Math.max(data.points.length - 1, 1)) * (data.width - data.pad * 2);
      const span = data.max - data.min || 1;
      const yP = data.height - data.pad - ((p.portfolio_return_pct - data.min) / span) * (data.height - data.pad * 2);
      const yB = data.height - data.pad - ((p.benchmark_return_pct - data.min) / span) * (data.height - data.pad * 2);
      
      hover.style.display = 'block';
      line.setAttribute('x1', x);
      line.setAttribute('x2', x);
      dotP.setAttribute('cx', x);
      dotP.setAttribute('cy', yP);
      dotB.setAttribute('cx', x);
      dotB.setAttribute('cy', yB);
      
      document.getElementById(`txt-date-${chartId}`).textContent = p.date;
      document.getElementById(`txt-p-${chartId}`).textContent = `Portfolio: ${p.portfolio_return_pct.toFixed(2)}%`;
      document.getElementById(`txt-b-${chartId}`).textContent = `Benchmark: ${p.benchmark_return_pct.toFixed(2)}%`;
      
      let tx = x + 10;
      if (tx + 150 > data.width) tx = x - 160;
      let ty = Math.min(yP, yB) - 70;
      if (ty < 10) ty = Math.max(yP, yB) + 10;
      tooltip.setAttribute('transform', `translate(${tx}, ${ty})`);
    }
    
    function hidePerfTooltip(chartId) {
      const hover = document.getElementById(`hover-${chartId}`);
      if (hover) hover.style.display = 'none';
    }

    async function changePeriod(period) {
      document.querySelectorAll('.timerange-picker button').forEach(b => b.classList.remove('active'));
      document.getElementById('btn-' + period).classList.add('active');
      await refreshPerformanceTimeRange(period);
    }

    async function refresh() {
      const response = await fetch('/api/latest', { cache: 'no-store' });
      const payload = await response.json();
      const run = payload.latest_run;
      const active = payload.active_run_id && (!run || payload.active_run_id !== run.run_id) ? ` | Active log: ${payload.active_run_id}` : '';
      document.getElementById('status').textContent = run ? `Latest: ${run.run_id}${active}` : `No completed runs found${active}`;
      if (!run) return;
      const summary = run.summary || {};
      document.getElementById('metrics').innerHTML = [
        ['Selected', summary.selected_symbols],
        ['Active decisions', summary.active_decisions],
        ['Planned orders', summary.planned_orders],
        ['Elapsed', summary.elapsed_human || ''],
      ].map(([label, value]) => `<div class="metric">${label}<strong>${fmt(value)}</strong></div>`).join('');
      document.getElementById('selected').innerHTML = table([
        ['Symbol', r => r.symbol],
        ['Score', r => Number((r.score_breakdown || {}).total || 0).toFixed(2)],
        ['Tradeable', r => r.is_tradeable],
        ['Flags', r => (r.data_quality_flags || []).join(', ')],
      ], run.selected_symbols || []);
      document.getElementById('decisions').innerHTML = table([
        ['Symbol', r => r.symbol],
        ['Action', r => r.action],
        ['Confidence', r => Number(r.confidence || 0).toFixed(2)],
        ['Allocation', r => Number(r.allocation || 0).toFixed(2)],
        ['R/R', r => r.reward_risk_ratio || ''],
      ], run.decisions || []);
      document.getElementById('orders').innerHTML = table([
        ['Symbol', r => r.symbol],
        ['Side', r => r.side],
        ['Qty', r => r.qty],
        ['Entry', r => r.entry_limit_price || ''],
        ['Stop', r => r.stop_price || ''],
        ['Target', r => r.take_profit_price || ''],
        ['Risk $', r => Number(r.risk_notional || 0).toFixed(2)],
      ], run.order_plans || []);
      document.getElementById('execution').innerHTML = table([
        ['Symbol', r => r.symbol],
        ['Side', r => r.side],
        ['Qty', r => r.qty],
        ['Status', r => r.status],
      ], run.execution_results || []);
      document.getElementById('logs').textContent = (payload.latest_log_tail || []).join('\n');
    }

    async function refreshPerformanceInception() {
      const response = await fetch('/api/performance?period=max', { cache: 'no-store' });
      const payload = await response.json();
      document.getElementById('performance-inception').innerHTML = renderPerformance(payload, 'inception');
      
      if (payload && !payload.error) {
        document.getElementById('portfolio-stats').innerHTML = [
          ['Total Trading Days', payload.total_days],
          ['Net Negative Days', payload.negative_days],
          ['Underperformed ' + (payload.benchmark_symbol || 'SPY'), payload.underperform_days],
          ['Win Rate', payload.total_days ? ((payload.total_days - payload.negative_days) / payload.total_days * 100).toFixed(1) + '%' : '0%'],
        ].map(([label, value]) => `<div class="metric">${label}<strong>${fmt(value)}</strong></div>`).join('');
      }
    }

    async function refreshPerformanceTimeRange(period = 'max') {
      const response = await fetch('/api/performance?period=' + period, { cache: 'no-store' });
      const payload = await response.json();
      document.getElementById('performance-timerange').innerHTML = renderPerformance(payload, 'timerange');
    }

    refresh().catch(err => { document.getElementById('status').textContent = err.message; });
    refreshPerformanceInception().catch(err => { document.getElementById('performance-inception').innerHTML = `<p class="muted">${err.message}</p>`; });
    refreshPerformanceTimeRange('max').catch(err => { document.getElementById('performance-timerange').innerHTML = `<p class="muted">${err.message}</p>`; });
    
    setInterval(() => refresh().catch(console.error), 10000);
    setInterval(() => {
      refreshPerformanceInception().catch(console.error);
      const activePeriod = document.querySelector('.timerange-picker button.active').id.replace('btn-', '');
      refreshPerformanceTimeRange(activePeriod).catch(console.error);
    }, 60000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local AI trading dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TradingConfig()
    server = DashboardServer(
        (args.host, args.port),
        DashboardRequestHandler,
        run_dir=config.run_dir,
        log_dir=config.log_dir,
        config=config,
    )
    print(f"Dashboard listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
