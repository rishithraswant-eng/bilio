set -euo pipefail
OUT_DIR=$PWD/out; mkdir -p $OUT_DIR; LOG=$OUT_DIR/run.log; : > $LOG
log(){ echo "$*" >> $LOG; }; fail(){ echo "FAIL: $*"; exit 1; }
judge_preflight(){ :; }; limit_dir(){ echo "$PWD/data"; }
PY=python3; USE_LLM=0; FORCE=0; OFFLINE_TEXT=1; PROVIDER=bilio; REQUIRE_JUDGE=0; ROOT_DIR=$PWD; V3=$PWD; AGENT_PID=
mkdir -p livekit_agent
cat > livekit_agent/fdb_v3_offline_replay.py <<PYX
import sys, json, pathlib, os
d = pathlib.Path(sys.argv[sys.argv.index("--data")+1]); p = sys.argv[sys.argv.index("--provider")+1]
mode = os.environ.get("STUB", "ok")
for i, e in enumerate(sorted(d.iterdir())):
    if mode == "partial" and i == 1: continue
    bad = {"bad": "timeout", "inference_failed": "inference_failed", "no_output": "no_output", "nostatus": None}
    st = bad[mode] if mode in bad and i == 0 else "completed"
    (e / f"result_{p}.json").write_text(json.dumps({"status": st} if st else {"actual_tool_calls": []}))
sys.exit(3 if mode == "crash" else 0)
PYX
cat > evaluate_tool_calls.py <<PYX
PYX
cp evaluate_tool_calls.py evaluate_pass_rate.py
source ./stage5.sh
stage_5_evaluate && echo "PASSED n=$(ls data/*/result_* | wc -l)"
