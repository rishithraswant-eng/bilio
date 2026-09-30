import json
import os
import sys
from pathlib import Path

fdb_v3_repo = Path("livekit_agent/.fdb_v3_repo/v3/benchmark_data_v2.json")
data_dir = Path("livekit_agent/.fdb_v3_data")

if not fdb_v3_repo.exists():
    print("benchmark_data_v2.json not found")
    sys.exit(1)

data_dir.mkdir(parents=True, exist_ok=True)

with open(fdb_v3_repo, "r", encoding="utf-8") as f:
    data = json.load(f)

for scenario in data.get("scenarios", []):
    scenario_id = scenario["id"]
    scenario_dir = data_dir / scenario_id
    scenario_dir.mkdir(exist_ok=True)
    
    # The offline evaluator expects metadata.json with the full scenario data
    # (specifically the "id" and "dialogue" fields)
    with open(scenario_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)

    # Creating a dummy input.wav just in case any shell scripts check for it
    with open(scenario_dir / "input.wav", "wb") as f:
        f.write(b"dummy wav data")

print(f"Created {len(data.get('scenarios', []))} dummy scenarios in {data_dir}")
