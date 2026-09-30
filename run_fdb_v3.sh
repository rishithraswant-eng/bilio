#!/usr/bin/env bash
# run_fdb_v3.sh — ONE command that reproduces the official FDB-v3 evaluation of the TriageLine agent.
#
#   ./run_fdb_v3.sh                     full run: install -> configure -> fetch -> agent -> inference -> 3 evaluators
#   ./run_fdb_v3.sh --limit 5           smoke run on the first 5 examples
#   ./run_fdb_v3.sh --offline-text      no LiveKit/keys needed: official data + official evaluators, text replay
#   ./run_fdb_v3.sh --skip-install      reuse the current environment
#
# Stages (each one is idempotent and fails fast with an actionable message):
#   1  venv (Python 3.10-3.12) + pinned deps (requirements-fdb.txt) + NeMo ASR; check ffmpeg
#   2  credentials: LIVEKIT_URL/KEY/SECRET, STT/TTS provider key(s), judge key -> v3/.env.local
#   3  clone FDB-v3 pinned to FDB_COMMIT; download + unzip the official data release (gdown)
#   4  copy official mock_apis/latency_injector; start livekit_agent/cascaded_agent.py; wait for registration
#   5  official inference (run_tool_benchmark_all_released.py) + evaluate_tool_calls / evaluate_pass_rate /
#      analyze_tool_latency, all with --provider $PROVIDER (LLM judge on unless --no-llm-judge)
#   6  copy reports, result_$PROVIDER.json files, agent log, pip freeze and SHAs to results/<timestamp>/,
#      then regenerate results/results.md
#
# Model/provider declaration: custom LiveKit agent (rules + optional schema-validated LLM planner via the
# provider chain in agent/providers.py). The effective STT/TTS/LLM settings are resolved by the same Python
# code the worker uses and written to results/<timestamp>/run_config.json. See README "Models used".
#
# Failure policy: the run FAILS (non-zero exit) when the official runner fails, when any expected example is
# missing or errored, or when --require-judge is set and the judge is not used. Result files from earlier
# runs are removed before inference so an old result can never be scored as a fresh one.
set -euo pipefail
export PYTHON="/c/Users/Rishith Raswant/AppData/Local/Programs/Python/Python312/python.exe"
export PYTHONUTF8=1

# ------------------------------------------------------------------------------------------ config
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LK_DIR="$ROOT_DIR/livekit_agent"
FDB_REPO_URL="https://github.com/DanielLin94144/Full-Duplex-Bench"
FDB_COMMIT="${FDB_COMMIT:-3e799c45a045256f47d5f1c9cda90157e2d2ec9e}"   # pinned (HEAD on 2026-05-20)
FDB_GDRIVE_ID="1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"                      # official data release (v3 README)
FDB_CLONE="$LK_DIR/.fdb_v3_repo"                                       # gitignored
V3="$FDB_CLONE/v3"
DATA_DIR="$V3/fdb_v3_data_released"
PROVIDER="${PROVIDER:-triageline}"                                     # -> result_triageline.json (B-03)
VENV="${VENV:-$ROOT_DIR/.venv-fdb}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$ROOT_DIR/results/$STAMP"
AGENT_LOG="$OUT_DIR/agent.log"
LOG="$OUT_DIR/run.log"

LIMIT=0; SKIP_INSTALL=0; SKIP_NEMO=0; USE_LLM=1; OFFLINE_TEXT=0; LATENCY="instant"; FORCE=0; REQUIRE_JUDGE=0
usage() { sed -n '2,20p' "$0"; cat <<'EOF'
Options:
  --limit N          only the first N examples (smoke run)
  --skip-install     do not create a venv / pip install (use the active python)
  --skip-nemo        do not install nemo_toolkit[asr] (only valid with --offline-text)
  --no-llm-judge     run the evaluators without --use-llm (exact-match lower bound)
  --offline-text     skip LiveKit: replay the official transcripts through the same adapter/agent
                     and score with the official evaluators (diagnostic, no keys needed)
  --latency P        mock API latency profile passed to the agent (instant|normal|slow), default instant
  --force            re-run examples that already have result files
  --require-judge    fail loudly if the gpt-4o judge is unreachable (never silently drop to exact match)
EOF
}
while [ $# -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-nemo) SKIP_NEMO=1; shift ;;
        --no-llm-judge) USE_LLM=0; shift ;;
        --offline-text) OFFLINE_TEXT=1; SKIP_NEMO=1; shift ;;
        --latency) LATENCY="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --require-judge) REQUIRE_JUDGE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1"; usage; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
log()  { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { log "BLOCKER: $*"; log "Stopped. Full log: $LOG"; exit 1; }

AGENT_PID=""
cleanup() {
    if [ -n "$AGENT_PID" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
        log "stopping agent (pid $AGENT_PID)"
        kill "$AGENT_PID" 2>/dev/null || true
        sleep 1; kill -9 "$AGENT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------------------ stage 1
stage_1_install() {
    log "=== Stage 1/6: environment + pinned dependencies ==="
    command -v ffmpeg >/dev/null || fail "ffmpeg not found (the official runner converts audio with it). Install: apt-get install -y ffmpeg / brew install ffmpeg"
    if [ "$SKIP_INSTALL" = 1 ]; then
        PY="${PYTHON:-$(command -v python3)}"
        log "--skip-install: using $PY ($("$PY" -V 2>&1))"
    else
        local base=""
        for c in ${PYTHON:-} python3.12 python3.11 python3.10; do command -v "$c" >/dev/null && { base="$c"; break; }; done
        [ -n "$base" ] || fail "need Python 3.10-3.12 (FDB-v3 targets 3.10; livekit-agents 1.8.3 supports <=3.14 but NeMo is tested on <=3.12). Install python3.12, or pass --skip-install to use the current interpreter."
        if [ ! -x "$VENV/bin/python" ]; then
            log "creating venv $VENV with $base"
            "$base" -m venv "$VENV" >>"$LOG" 2>&1 || fail "venv creation failed"
        fi
        PY="$VENV/bin/python"
        "$PY" -m pip install -q --upgrade pip >>"$LOG" 2>&1
        log "pip install -r requirements-fdb.txt"
        "$PY" -m pip install -q -r "$ROOT_DIR/requirements-fdb.txt" >>"$LOG" 2>&1 || fail "pip install failed, see $LOG"
        "$PY" -m pip install -q pytest >>"$LOG" 2>&1 || true
    fi
    local pymax="(3,12)"; [ "$SKIP_INSTALL" = 1 ] && [ "$SKIP_NEMO" = 1 ] && pymax="(3,14)"
    "$PY" -c "import sys; assert (3,10) <= sys.version_info[:2] <= $pymax, sys.version" \
        || fail "Python $("$PY" -V) is outside the supported range (3.10-3.12; NeMo ASR used by the official runner is tested <=3.12)"
    if [ "$SKIP_NEMO" = 0 ]; then
        if "$PY" -c "import nemo.collections.asr" >/dev/null 2>&1; then
            log "NeMo ASR already installed"
        else
            log "installing nemo_toolkit[asr]==2.5.3 (large; used by the official runner for output ASR)"
            "$PY" -m pip install -q "nemo_toolkit[asr]==2.5.3" >>"$LOG" 2>&1 || fail "nemo_toolkit install failed, see $LOG"
        fi
    fi
    "$PY" -c "import livekit.agents, dotenv, openai, gdown" 2>>"$LOG" || fail "core packages not importable"
    if [ "$OFFLINE_TEXT" = 0 ] && [ "${TRIAGELINE_TURN_DETECTOR:-1}" = 1 ]; then
        # semantic turn-detector / Silero weights must be on disk BEFORE the first room (clean machine)
        log "downloading LiveKit plugin model weights (turn detector, silero)"
        ( cd "$LK_DIR" && "$PY" -m livekit.agents download-files ) >>"$LOG" 2>&1 \
            || { log "WARNING: download-files failed -> TRIAGELINE_TURN_DETECTOR=0 (VAD endpointing only)"; export TRIAGELINE_TURN_DETECTOR=0; }
    fi
    log "python: $("$PY" -V 2>&1); livekit-agents $("$PY" -c 'import livekit.agents as a; print(a.__version__)')"
}

# ------------------------------------------------------------------------------------------ stage 2
stage_2_configure() {
    log "=== Stage 2/6: credentials ==="
    # load livekit_agent/.env.local if present (never printed, never copied to results/)
    # repo-root .env first, then livekit_agent/.env.local (loaded last, so it wins)
    local f
    for f in "$ROOT_DIR/.env" "$LK_DIR/.env.local"; do
        if [ -f "$f" ]; then set -a; . "$f"; set +a; log "loaded ${f#$ROOT_DIR/}"; fi
    done
    # scored configuration, pinned (reproducible): rules-only benchmark agent, rules-first LLM mode
    export TRIAGELINE_MODE=benchmark TRIAGELINE_BENCHMARK_POLICY=1
    export TRIAGELINE_LLM_PLANNER="${TRIAGELINE_LLM_PLANNER_OVERRIDE:-0}" TRIAGELINE_LLM_MODE=fallback
    if [ "$OFFLINE_TEXT" = 1 ]; then
        [ "$USE_LLM" = 0 ] || [ -n "${OPENAI_API_KEY:-}" ] || { log "no OPENAI_API_KEY: judge disabled (exact match)"; USE_LLM=0; }
        return 0
    fi
    local missing=()
    for v in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do [ -n "${!v:-}" ] || missing+=("$v"); done
    # Use the exact same resolver as dev/console; avoid conflicting shell defaults.
    local providers
    providers=$(cd "$ROOT_DIR" && "$PY" - <<'PYCONFIG'
from livekit_agent.speech_providers import selected
c = selected()
if c["missing_keys"]:
    raise SystemExit("missing speech credentials: " + ", ".join(c["missing_keys"]))
print(c["stt_provider"], c["tts_provider"])
PYCONFIG
    ) || fail "invalid speech provider configuration"
    local stt tts
    read -r stt tts <<< "$providers"
    export TRIAGELINE_STT_PROVIDER="$stt" TRIAGELINE_TTS_PROVIDER="$tts"
    if [ "$USE_LLM" = 1 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        [ "$REQUIRE_JUDGE" = 0 ] || fail "--require-judge needs OPENAI_API_KEY for the official gpt-4o judge"
        log "no OPENAI_API_KEY: judge disabled (exact match; not the official judged score)"
        USE_LLM=0
    fi
    [ ${#missing[@]} -eq 0 ] || fail "missing: ${missing[*]}. Copy livekit_agent/.env.example to livekit_agent/.env.local and fill it in (see docs/FREE_API_KEYS.md)."
    # a key being present is not enough: prove the exact STT/TTS model names work BEFORE 100 silent examples
    if [ "${SKIP_SPEECH_PROBE:-0}" != 1 ]; then
        local probe="$OUT_DIR/speech_preflight.txt"
        ( cd "$ROOT_DIR" && "$PY" scripts/check_providers.py --no-llm --speech ) >"$probe" 2>&1 && local prc=0 || local prc=$?
        cat "$probe" >>"$LOG"; grep -E "round trip|FAIL" "$probe" || true
        [ "$prc" = 0 ] || fail "speech preflight failed (TTS->STT round trip with STT=$stt TTS=$tts). Fix the key / model name (TRIAGELINE_STT_MODEL, TRIAGELINE_TTS_MODEL) or set SKIP_SPEECH_PROBE=1 to bypass"
    fi
    log "credentials present (STT=$stt, TTS=$tts, judge=$([ "$USE_LLM" = 1 ] && echo gpt-4o || echo off))"
}

write_v3_env() {
    # the official runner + livekit_inference.py load v3/.env.local
    umask 077
    {
        echo "LIVEKIT_URL=${LIVEKIT_URL:-}"
        echo "LIVEKIT_API_KEY=${LIVEKIT_API_KEY:-}"
        echo "LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET:-}"
        echo "OPENAI_API_KEY=${OPENAI_API_KEY:-}"
        [ -n "${OPENAI_BASE_URL:-}" ] && echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
        echo "LK_PROVIDER=$PROVIDER"
    } > "$V3/.env.local"
}

# ------------------------------------------------------------------------------------------ stage 3
stage_3_fetch() {
    log "=== Stage 3/6: official FDB-v3 code (pinned $FDB_COMMIT) + data ==="
    if [ ! -d "$FDB_CLONE/.git" ]; then
        rm -rf "$FDB_CLONE"
        git clone -q "$FDB_REPO_URL" "$FDB_CLONE" >>"$LOG" 2>&1 || fail "git clone $FDB_REPO_URL failed"
    fi
    git -C "$FDB_CLONE" fetch -q origin "$FDB_COMMIT" >>"$LOG" 2>&1 || true
    git -C "$FDB_CLONE" checkout -q "$FDB_COMMIT" >>"$LOG" 2>&1 || fail "could not check out pinned commit $FDB_COMMIT"
    log "FDB-v3 at $(git -C "$FDB_CLONE" rev-parse HEAD)"
    if [ -d "$DATA_DIR" ] && [ "$(find "$DATA_DIR" -maxdepth 2 -name input.wav | wc -l)" -ge 100 ]; then
        log "data already present: $DATA_DIR"
    else
        local zip="$FDB_CLONE/fdb_v3_data_released.zip" tmp="$FDB_CLONE/.unzip"
        [ -s "$zip" ] || { log "downloading official data release (~736 MB) via gdown"; \
            "$PY" -m gdown -q "$FDB_GDRIVE_ID" -O "$zip" >>"$LOG" 2>&1 || fail "gdown download failed (Drive quota?). Download it manually from the v3 README link and unzip to $DATA_DIR"; }
        rm -rf "$tmp"; mkdir -p "$tmp"
        "$PY" -m zipfile -e "$zip" "$tmp" >>"$LOG" 2>&1 || fail "unzip failed"
        rm -rf "$tmp/__MACOSX" "$DATA_DIR"
        mv "$(find "$tmp" -maxdepth 2 -type d -name fdb_v3_data_released | head -1)" "$DATA_DIR" || fail "unexpected zip layout"
        rm -rf "$tmp" "$zip"
        log "data ready: $(find "$DATA_DIR" -maxdepth 2 -name input.wav | wc -l) recordings"
    fi
}

# ------------------------------------------------------------------------------------------ stage 4
stage_4_agent() {
    log "=== Stage 4/6: start the TriageLine LiveKit agent ==="
    cp "$V3/mock_apis.py" "$V3/latency_injector.py" "$LK_DIR/"      # official backend, unmodified
    write_v3_env
    : > /tmp/agent_heartbeat.log; : > /tmp/agent_tool_calls.log     # same fixed paths as upstream
    ( cd "$LK_DIR" && exec "$PY" cascaded_agent.py start --latency "$LATENCY" ) >"$AGENT_LOG" 2>&1 &
    AGENT_PID=$!
    local waited=0
    until grep -qiE "registered worker|worker registered" "$AGENT_LOG" 2>/dev/null; do
        kill -0 "$AGENT_PID" 2>/dev/null || fail "agent exited during start-up, see $AGENT_LOG"
        [ $waited -lt 120 ] || fail "agent did not register with LiveKit within 120 s, see $AGENT_LOG"
        sleep 1; waited=$((waited + 1))
    done
    log "agent registered (pid $AGENT_PID) after ${waited}s"
}

# ------------------------------------------------------------------------------------------ stage 5
limit_dir() {
    # --limit: evaluate a subset by pointing the runner at a directory of symlinks
    if [ "$LIMIT" -gt 0 ]; then
        local sub="$FDB_CLONE/subset_$LIMIT"
        rm -rf "$sub"; mkdir -p "$sub"
        # real directories with hard-linked files: the official evaluators use rglob(), which does not
        # descend into symlinked directories; hard links cost no extra disk space
        for d in $(ls "$DATA_DIR" | sort | head -n "$LIMIT"); do
            mkdir -p "$sub/$d"
            for f in "$DATA_DIR/$d"/metadata.json "$DATA_DIR/$d"/input.wav; do
                [ -e "$f" ] && { ln "$(readlink -f "$f")" "$sub/$d/" 2>/dev/null || cp "$f" "$sub/$d/"; }
            done
        done
        echo "$sub"
    else
        echo "$DATA_DIR"
    fi
}

judge_preflight() {
    # The official evaluators silently fall back to exact match (arguments) and score 0 (response quality)
    # when gpt-4o is unreachable. Check once, up front, so a report never mixes judged and unjudged numbers.
    [ "$USE_LLM" = 1 ] || return 0
    if "$PY" - >>"$LOG" 2>&1 <<'PYEOF'
from openai import OpenAI
r = OpenAI().chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Reply with the word ok"}],
                                     max_tokens=3, temperature=0)
assert r.choices[0].message.content
PYEOF
    then
        log "LLM judge preflight: gpt-4o reachable${OPENAI_BASE_URL:+ via \$OPENAI_BASE_URL}"
    else
        [ "$REQUIRE_JUDGE" = 1 ] && fail "gpt-4o judge NOT reachable and --require-judge was given (check OPENAI_API_KEY/OPENAI_BASE_URL)"
        log "WARNING: gpt-4o judge NOT reachable with the configured OPENAI_API_KEY/OPENAI_BASE_URL -> running WITHOUT --use-llm (exact match, lower bound). See docs/FREE_API_KEYS.md"
        USE_LLM=0
    fi
}

stage_5_evaluate() {
    log "=== Stage 5/6: official inference + evaluators (provider=$PROVIDER) ==="
    judge_preflight
    local data; data="$(limit_dir)"
    local judge=(); [ "$USE_LLM" = 1 ] && judge=(--use-llm)
    local force=(); [ "$FORCE" = 1 ] && force=(--force)
    [ "$OFFLINE_TEXT" = 0 ] || [ "$PROVIDER" != triageline ] || PROVIDER="triageline_text"   # never mix with live results
    local expected; expected=$(find "$data" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$expected" -gt 0 ] || fail "no examples found in $data"
    if [ "$FORCE" = 1 ] || [ "${KEEP_OLD_RESULTS:-0}" != 1 ]; then
        # isolate this run: stale result files from an earlier (possibly interrupted) run must never be scored
        find "$data" -name "result_${PROVIDER}.json" -delete
        find "$data" \( -name "result_${PROVIDER}*.wav" -o -name "output_${PROVIDER}*.wav" \) -delete 2>/dev/null || true
        force=(--force)
    fi
    local marker="$OUT_DIR/.inference_started"; : > "$marker"
    local rc=0
    if [ "$OFFLINE_TEXT" = 1 ]; then
        log "offline text replay (no LiveKit transport, no audio latency) -> result_${PROVIDER}.json"
        ( cd "$ROOT_DIR" && "$PY" livekit_agent/fdb_v3_offline_replay.py --data "$data" --text --provider "$PROVIDER" ) \
            >>"$LOG" 2>&1 || rc=$?
    else
        # pipefail: the runner's exit status survives tee; grep only filters the console view
        ( cd "$V3" && "$PY" run_tool_benchmark_all_released.py --provider "$PROVIDER" --root_dir "$data" "${force[@]}" ) \
            > >(tee -a "$LOG" | grep --line-buffered -E "Processing|Saved|failed|error" || true) 2>&1 || rc=$?
        kill -0 "$AGENT_PID" 2>/dev/null || fail "the TriageLine agent died during inference, see $AGENT_LOG"
    fi
    [ "$rc" = 0 ] || fail "inference exited with status $rc (see $LOG); refusing to score a partial run"
    local n bad
    n=$(find "$data" -name "result_${PROVIDER}.json" -newer "$marker" | wc -l)
    bad=$("$PY" - "$data" "$PROVIDER" <<'PYCHK'
import json, pathlib, sys
root, prov = pathlib.Path(sys.argv[1]), sys.argv[2]
bad = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    f = d / f"result_{prov}.json"
    try:
        data = json.loads(f.read_text())
        # only an explicitly completed run counts: the official runner also writes
        # "inference_failed" / "no_output" results, which would silently score as zero tool calls
        if not isinstance(data, dict) or data.get("error") or str(data.get("status", "")).lower() != "completed":
            bad.append(f"{d.name}({data.get('status') if isinstance(data, dict) else type(data).__name__})")
    except Exception:
        bad.append(d.name)
print(" ".join(bad))
PYCHK
)
    [ -z "$bad" ] || fail "missing/errored results for: $bad"
    [ "$n" -eq "$expected" ] || fail "expected $expected fresh result files, found $n"
    if [ "$REQUIRE_JUDGE" = 1 ] && [ "$USE_LLM" != 1 ]; then fail "--require-judge given but the judge is disabled"; fi
    log "$n/$expected fresh result files; running evaluators ${judge[*]:-(exact match)}"
    ( cd "$V3" && "$PY" evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$data" \
        --provider "$PROVIDER" --output "$OUT_DIR/${PROVIDER}_evaluation_report.json" "${judge[@]}" ) >>"$LOG" 2>&1 \
        || fail "evaluate_tool_calls.py failed"
    ( cd "$V3" && "$PY" evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$data" \
        --provider "$PROVIDER" --output "$OUT_DIR/${PROVIDER}_pass_rate_report.json" "${judge[@]}" ) >>"$LOG" 2>&1 \
        || fail "evaluate_pass_rate.py failed"
    if [ "$USE_LLM" = 1 ]; then
        # a passing preflight does not prove every item was judged (the evaluators fall back per item)
        if "$PY" "$ROOT_DIR/scripts/judge_coverage.py" "$OUT_DIR/${PROVIDER}_evaluation_report.json" \
                "$OUT_DIR/${PROVIDER}_pass_rate_report.json" 2>&1 | tee -a "$LOG" | head -5; then :; fi
        if ! "$PY" "$ROOT_DIR/scripts/judge_coverage.py" "$OUT_DIR/${PROVIDER}_evaluation_report.json" \
                "$OUT_DIR/${PROVIDER}_pass_rate_report.json" >/dev/null 2>&1; then
            [ "$REQUIRE_JUDGE" = 1 ] && fail "some checks were NOT judged by gpt-4o (see judge coverage above); re-run with a working judge"
            log "WARNING: some checks fell back to exact match (see judge coverage above)"
        fi
    fi
    if [ "$OFFLINE_TEXT" = 0 ]; then
        if [ -n "${OPENAI_API_KEY:-}" ]; then
            ( cd "$V3" && "$PY" analyze_tool_latency.py --results-dir "$data" --provider "$PROVIDER" \
                --output "$OUT_DIR/${PROVIDER}_latency_report.json" ) >>"$LOG" 2>&1 || log "WARNING: analyze_tool_latency.py failed (see log)"
        else
            log "analyze_tool_latency.py skipped (needs OPENAI_API_KEY for its gpt-4o step)"
        fi
    fi
    EVAL_DATA="$data"
}

# ------------------------------------------------------------------------------------------ stage 6
stage_6_save() {
    log "=== Stage 6/6: save artefacts + summary ==="
    mkdir -p "$OUT_DIR/per_example"
    find "$EVAL_DATA" -name "result_${PROVIDER}.json" | while read -r f; do
        cp "$f" "$OUT_DIR/per_example/$(basename "$(dirname "$f")").json"
    done
    "$PY" -m pip freeze > "$OUT_DIR/pip_freeze.txt" 2>/dev/null || true
    cp /tmp/agent_heartbeat.log "$OUT_DIR/agent_heartbeat.log" 2>/dev/null || true
    cp /tmp/agent_tool_calls.log "$OUT_DIR/agent_tool_calls.log" 2>/dev/null || true
    # effective settings resolved by the same code the worker runs (not shell defaults)
    ( cd "$ROOT_DIR" && STAMP="$STAMP" PROVIDER="$PROVIDER" LIMIT="$LIMIT" LATENCY="$LATENCY" \
      MODE="$([ "$OFFLINE_TEXT" = 1 ] && echo offline_text_replay || echo livekit_live)" \
      JUDGE="$USE_LLM" FDB_SHA="$(git -C "$FDB_CLONE" rev-parse HEAD 2>/dev/null)" \
      TL_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null)" \
      TL_DIRTY="$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null | grep -v '^??' | wc -l)" \
      "$PY" - > "$OUT_DIR/run_config.json" <<'PYCFG'
import json, os, sys
sys.path.insert(0, ".")
os.environ.setdefault("TRIAGELINE_BENCHMARK_POLICY", "1")
from agent import llm_planner
try:
    from livekit_agent.speech_providers import selected
    sp = selected()
except Exception as exc:  # offline text mode may not need speech providers
    sp = {"error": type(exc).__name__}
e = os.environ
print(json.dumps({
    "timestamp_utc": e["STAMP"], "provider": e["PROVIDER"], "mode": e["MODE"], "limit": int(e["LIMIT"]),
    "llm_judge": e["JUDGE"] == "1", "latency_profile": e["LATENCY"],
    "fdb_commit": e["FDB_SHA"], "triageline_commit": e["TL_SHA"], "triageline_dirty_files": int(e["TL_DIRTY"]),
    "python": sys.version.split()[0],
    "speech": {k: sp.get(k) for k in ("stt_provider", "stt_model", "tts_provider", "tts_model", "tts_voice", "error") if k in sp},
    "stt_bias": e.get("TRIAGELINE_STT_BIAS", "1"),
    "settle_s": e.get("TRIAGELINE_SETTLE_S", "1.0"), "max_settle_s": e.get("TRIAGELINE_MAX_SETTLE_S", "2.0"),
    "benchmark_policy": e.get("TRIAGELINE_BENCHMARK_POLICY"),
    "triageline_mode": e.get("TRIAGELINE_MODE", "benchmark"),
    "llm_mode": e.get("TRIAGELINE_LLM_MODE", "fallback"), "llm_planner_setting": e.get("TRIAGELINE_LLM_PLANNER", "auto"),
    "turn_detector": e.get("TRIAGELINE_TURN_DETECTOR", "1"),
    "llm_planner": {"enabled": llm_planner.enabled(), **{k: v for k, v in llm_planner.status().items() if k != "enabled"}},
    "seeds": {"mock_latency_profile": e["LATENCY"], "stt_temperature": 0, "llm_temperature": 0, "llm_seed": 7},
}, indent=2, default=str))
PYCFG
    ) || log "WARNING: could not resolve effective config"
    "$PY" "$ROOT_DIR/results/summarize_fdb_v3.py" "$OUT_DIR" | tee -a "$LOG"
    log "done. Artefacts: $OUT_DIR   Summary: results/results.md"
}

stage_1_install
stage_2_configure
stage_3_fetch
[ "$OFFLINE_TEXT" = 1 ] || stage_4_agent
stage_5_evaluate
stage_6_save
