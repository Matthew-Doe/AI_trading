from __future__ import annotations

import os
from pathlib import Path
from trading_system.utils import read_json, get_logger
from trading_system.main import (
    symbol_market_data_from_dict,
    symbol_debate_from_dict,
    trade_decision_from_dict,
)
from trading_system.reporting import write_ai_debug_log

def rebuild():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No runs directory found.")
        return

    # Mock logger for the utility functions
    logger = get_logger(Path("logs"), "rebuild_debug_logs")

    run_paths = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    print(f"Found {len(run_paths)} runs. Starting rebuild...")

    for run_path in run_paths:
        try:
            # Check for required files
            reqs = ["selected_symbols.json", "debates.json", "decisions.json", "execution_results.json"]
            if not all((run_path / f).exists() for f in reqs):
                print(f"Skipping {run_path.name}: missing files.")
                continue

            # Load data
            selected_symbols = [
                symbol_market_data_from_dict(item) 
                for item in read_json(run_path / "selected_symbols.json")
            ]
            debates = [
                symbol_debate_from_dict(item) 
                for item in read_json(run_path / "debates.json")
            ]
            decisions = [
                trade_decision_from_dict(item) 
                for item in read_json(run_path / "decisions.json")
            ]
            execution_results = read_json(run_path / "execution_results.json")

            # Generate the debug log
            write_ai_debug_log(
                run_path=run_path,
                selected_symbols=selected_symbols,
                debates=debates,
                decisions=decisions,
                execution_results=execution_results
            )
            print(f"Rebuilt ai_debug_log.json for {run_path.name}")

        except Exception as e:
            print(f"Error rebuilding {run_path.name}: {e}")

if __name__ == "__main__":
    rebuild()
