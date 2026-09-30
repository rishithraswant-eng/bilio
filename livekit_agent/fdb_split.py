"""Deterministic dev / held-out split of the 79 FDB-v3 scenarios (anti-overfitting, audit §4).

A scenario id is assigned to "dev" or "heldout" by a SHA-256 hash of its id, so the split is
reproducible, independent of file order, and keeps both recordings of a scenario on the same side.
Rules for contributors: inspect and tune ONLY on dev; heldout numbers are reported, never debugged.

    python3 livekit_agent/fdb_split.py --data <fdb_v3_data_released>     # prints both lists
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def side(scenario_id: str) -> str:
    h = int(hashlib.sha256(("bilio-split-v1:" + scenario_id).encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "heldout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    ids = sorted({json.loads((p / "metadata.json").read_text())["id"]
                  for p in Path(a.data).iterdir() if (p / "metadata.json").exists()})
    for s in ("dev", "heldout"):
        chosen = [i for i in ids if side(i) == s]
        print(f"{s}: {len(chosen)} scenarios\n  " + " ".join(chosen))


if __name__ == "__main__":
    main()
