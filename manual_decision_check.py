from __future__ import annotations

import json
from pathlib import Path
from trading_system.config import TradingConfig
from trading_system.decision import DecisionEngine
from trading_system.main import symbol_debate_from_dict
from trading_system.utils import get_logger

def manual_check():
    config = TradingConfig()
    config.llm_provider = "ollama"
    logger = get_logger(Path("logs"), "manual_check")
    
    # Use data from a previous run to test
    run_dir = Path("backtests/20260425T224327Z/2026-04-20")
    if not run_dir.exists():
        # Fallback to older run
        run_dir = Path("backtests/20260425T044511Z/2026-04-20")
        
    debates_payload = json.load(open(run_dir / "debates.json"))
    debates = [symbol_debate_from_dict(item) for item in debates_payload]
    
    # Test just the first 3 symbols to save time
    test_debates = debates[:3]
    
    engine = DecisionEngine(config, logger)
    print(f"Running manual decision check for symbols: {[d.symbol for d in test_debates]}")
    
    decisions = engine.decide(test_debates)
    
    for d in decisions:
        print(f"Symbol: {d.symbol} | Action: {d.action} | Confidence: {d.confidence}")

if __name__ == "__main__":
    manual_check()
