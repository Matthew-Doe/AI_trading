from __future__ import annotations

import json
import math
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

def calculate_metrics(trades, daily_equity, benchmark_returns, risk_free_rate=0.04):
    """
    Calculates advanced performance and risk metrics.
    trades: list of trade dicts with return_pct, confidence, side, etc.
    daily_equity: list of daily portfolio returns (%)
    benchmark_returns: list of daily benchmark returns (%)
    """
    if not trades:
        return {}

    # Basic Trade Stats
    returns = [t['return_pct'] / 100 for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    win_rate = len(wins) / len(returns)
    
    # 1. EV (Expected Value)
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # 2. Profit Factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # 3. Sharpe & Sortino (Daily based)
    daily_rets = np.array(daily_equity) / 100
    excess_rets = daily_rets - (risk_free_rate / 252)
    sharpe = np.mean(excess_rets) / np.std(daily_rets) * math.sqrt(252) if np.std(daily_rets) > 0 else 0
    
    downside_rets = daily_rets[daily_rets < 0]
    sortino = np.mean(excess_rets) / np.std(downside_rets) * math.sqrt(252) if len(downside_rets) > 0 and np.std(downside_rets) > 0 else 0
    
    # 4. Max Drawdown
    cum_equity = np.cumprod(1 + daily_rets)
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns = (cum_equity - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    # 5. Calmar
    annual_ret = np.mean(daily_rets) * 252
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    
    # 6. Alpha & Information Ratio
    bench_rets = np.array(benchmark_returns) / 100
    if len(daily_rets) == len(bench_rets):
        # Simplistic Alpha: Excess return over benchmark
        alpha = (annual_ret - (np.mean(bench_rets) * 252))
        
        # Information Ratio
        tracking_error = np.std(daily_rets - bench_rets) * math.sqrt(252)
        info_ratio = alpha / tracking_error if tracking_error > 0 else 0
    else:
        alpha, info_ratio = 0, 0
        
    # 7. VaR & CVaR (95% confidence)
    var_95 = np.percentile(daily_rets, 5)
    cvar_95 = daily_rets[daily_rets <= var_95].mean() if len(daily_rets[daily_rets <= var_95]) > 0 else var_95
    
    # 8. Prediction Quality (Correlation between confidence and outcome)
    confidences = [t.get('confidence', 0) for t in trades]
    if len(confidences) > 1 and np.std(confidences) > 0:
        pred_quality = np.corrcoef(confidences, returns)[0, 1]
    else:
        pred_quality = 0

    return {
        "ev": round(ev * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "alpha_annual": round(alpha * 100, 2),
        "info_ratio": round(info_ratio, 2),
        "var_95_daily": round(var_95 * 100, 2),
        "cvar_95_daily": round(cvar_95 * 100, 2),
        "prediction_quality_corr": round(pred_quality, 3),
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1)
    }

def run_analysis(backtest_dir, allocation_per_trade=5000, slippage_pct=0.20):
    # 1. Collect all trade and decision data
    all_trades = []
    daily_equity_curve = []
    benchmark_rets = []
    
    # Find outcome files
    outcome_paths = sorted(glob.glob(f"{backtest_dir}/*/outcomes.json"))
    
    for opath in outcome_paths:
        day_dir = Path(opath).parent
        outcomes = json.load(open(opath))
        
        # Load decisions for confidence data
        dec_path = day_dir / "decisions.json"
        confidence_map = {}
        if dec_path.exists():
            decs = json.load(open(dec_path))
            for d in decs:
                confidence_map[d['symbol']] = d.get('confidence', 0)
        
        day_trades = outcomes.get('trades', [])
        net_day_ret_pct = 0
        
        for t in day_trades:
            # Inject confidence
            t['confidence'] = confidence_map.get(t['symbol'], 0)
            # Apply slippage
            t['net_return_pct'] = t['return_pct'] - slippage_pct
            all_trades.append(t)
            
            # Simple daily equity contribution (non-compounded for now)
            # contribution = (5k / 100k account) * return
            # Assume 100k account base for the % calc
            net_day_ret_pct += (allocation_per_trade / 100000) * t['net_return_pct']
            
        daily_equity_curve.append(net_day_ret_pct)
        
        # In a real scenario we'd pull actual SPY returns here. 
        # For this estimate, we'll use the 'benchmark_return_pct' from the dashboard API if we had it,
        # but since we are in a script, we'll proxy it or assume 0 for now until we add the yfinance hook.
        # Let's assume 0.05% avg daily return for SPY as a placeholder.
        benchmark_rets.append(0.05)

    stats = calculate_metrics(all_trades, daily_equity_curve, benchmark_rets)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    import sys
    latest_run = sorted(glob.glob("backtests/*"))[-1]
    print(f"Analyzing {latest_run}...")
    run_analysis(latest_run)
