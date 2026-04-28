import json
import numpy as np
from pathlib import Path


def analyze_report(report_path):
    with open(report_path) as f:
        data = json.load(f)

    confidence_analysis = data.get("confidence_analysis")
    if confidence_analysis:
        print(f"Total trades: {confidence_analysis['trade_count']}")
        print(f"Average return: {confidence_analysis['average_return_pct']:.4%}")
        print(f"Median return: {confidence_analysis['median_return_pct']:.4%}")
        print(f"Profit factor: {confidence_analysis['profit_factor']}")
        correlations = confidence_analysis.get("correlations", {})
        print(
            "Confidence-Return Correlation: "
            f"{correlations.get('confidence_vs_return_pct', 0.0):.4f}"
        )
        print(
            "Confidence-P/L Correlation: "
            f"{correlations.get('confidence_vs_net_pnl', 0.0):.4f}"
        )
        print(
            "Confidence-Win Correlation: "
            f"{correlations.get('confidence_vs_win_loss', 0.0):.4f}"
        )
        for label, bucket in confidence_analysis.get("buckets", {}).items():
            if bucket.get("trade_count"):
                print(
                    f"Bucket {label}: count={bucket['trade_count']}, "
                    f"win_rate={bucket['win_rate']:.1%}, "
                    f"avg_ret={bucket['average_return_pct']:.2%}, "
                    f"median_ret={bucket['median_return_pct']:.2%}, "
                    f"profit_factor={bucket['profit_factor']}"
                )
        return
    
    trades = data['all_trades']
    if not trades:
        print("No trades found.")
        return

    confidences = [t.get('confidence', 0.0) for t in trades]
    returns = [
        t.get('return_pct') or t['net_pnl'] / (t['entry_price'] * t['qty'])
        for t in trades
    ]
    
    print(f"Total trades: {len(trades)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Average return: {np.mean(returns):.4%}")
    
    if len(confidences) > 1:
        corr = np.corrcoef(confidences, returns)[0, 1]
        print(f"Confidence-Return Correlation: {corr:.4f}")
    
    # Analyze by confidence buckets
    buckets = np.linspace(0, 1, 11)
    for i in range(len(buckets)-1):
        lower = buckets[i]
        upper = buckets[i+1]
        bucket_trades = [r for c, r in zip(confidences, returns) if lower <= c < upper]
        if bucket_trades:
            win_rate = len([r for r in bucket_trades if r > 0]) / len(bucket_trades)
            avg_ret = np.mean(bucket_trades)
            print(f"Bucket {lower:.1f}-{upper:.1f}: count={len(bucket_trades)}, win_rate={win_rate:.1%}, avg_ret={avg_ret:.2%}")

    # Analyze by side
    for side in ['long', 'short']:
        side_trades = [t for t in trades if t['side'] == side]
        if side_trades:
            side_conf = [t['confidence'] for t in side_trades]
            side_ret = [t['net_pnl'] / (t['entry_price'] * t['qty']) for t in side_trades]
            win_rate = len([r for r in side_ret if r > 0]) / len(side_trades)
            print(f"Side {side}: count={len(side_trades)}, win_rate={win_rate:.1%}, avg_ret={np.mean(side_ret):.2%}, corr={np.corrcoef(side_conf, side_ret)[0,1]:.4f}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "backtests/20260426T040610Z/backtest_report.json"
    analyze_report(path)
