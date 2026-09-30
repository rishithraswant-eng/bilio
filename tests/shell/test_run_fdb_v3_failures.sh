#!/usr/bin/env bash
# Verifies run_fdb_v3.sh stage 5 refuses to score crashed, partial, errored or stale runs.
# Uses a stub runner; no network, LiveKit or benchmark data needed.   bash tests/shell/test_run_fdb_v3_failures.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
mkdir -p "$W/data/ex1" "$W/data/ex2"
python3 - "$ROOT/run_fdb_v3.sh" "$W/stage5.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
start = s.index("stage_5_evaluate() {")
open(sys.argv[2], "w").write(s[start:s.index("stage_6_save() {", start)])
PY
cp "$HERE/stage5_harness.sh" "$W/harness.sh"
cd "$W"
expect() { local mode=$1 want=$2 got; rm -f data/*/result_*; [ "$mode" = stale ] && echo '{"status":"completed"}' > data/ex2/result_bilio_text.json
  got=$(STUB=$([ "$mode" = stale ] && echo partial || echo "$mode") bash harness.sh 2>&1 | tail -1)
  case "$got" in "$want"*) echo "ok   $mode";; *) echo "FAIL $mode: $got"; exit 1;; esac; }
expect ok PASSED; expect crash FAIL; expect partial FAIL; expect bad FAIL; expect stale FAIL
# official runner failure statuses and a result with no status must never be scored as valid
expect inference_failed FAIL; expect no_output FAIL; expect nostatus FAIL
echo "all run_fdb_v3.sh failure-detection checks passed"
