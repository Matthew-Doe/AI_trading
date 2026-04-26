import json
import glob
from pathlib import Path

def verify():
    latest_run = sorted(glob.glob("backtests/*"))[-1]
    day_dirs = sorted(glob.glob(f"{latest_run}/20*"), reverse=True)
    if not day_dirs:
        print("No days processed yet.")
        return

    latest_day = day_dirs[0]
    dec_path = Path(latest_day) / "decisions.json"
    if not dec_path.exists():
        print(f"No decisions.json found in {latest_day}")
        return

    decs = json.load(open(dec_path))
    confs = [d['confidence'] for d in decs if d['action'] != 'skip']
    if not confs:
        print(f"No trades taken on {latest_day}")
        return

    unique_confs = set(confs)
    print(f"Verification for {latest_day}:")
    print(f"Total Trades: {len(confs)}")
    print(f"Unique Confidence Scores: {len(unique_confs)}")
    print(f"Scores: {confs}")
    
    if len(unique_confs) > 1:
        print("SUCCESS: Confidence scores are now varied!")
    else:
        print("STILL STALLED: All confidence scores are the same.")

if __name__ == "__main__":
    verify()
