#!/usr/bin/env python3
"""FDB-v3 OFFLINE REPLAY on the OFFICIAL released data, scored by the OFFICIAL evaluator.

What this is
------------
The official FDB-v3 pipeline streams input.wav into a LiveKit room and records the agent. This sandbox has
no LiveKit server / LIVEKIT_* credentials and the OpenAI proxy exposes no audio (STT/TTS) endpoints, so the
live room cannot be run here. This script replays the SAME official material through the SAME
competition path, minus the network transport:

    official input.wav (48 kHz, human-recorded)
      -> local faster-whisper ASR (segment timestamps; pauses >= GAP_S split user turns)
      -> TriageAdapter.on_user_final()            (livekit_agent/adapter.py — the LiveKit adapter)
      -> ParticipantAgent                         (agent/agent.py — epochs, cancellation, ledger)
      -> tool executor -> OFFICIAL mock_apis.MockAPIRegistry (cloned FDB-v3 repo, unmodified)
      -> result_<provider>.json in the official schema (actual_tool_calls, transcript, ...)
    then the OFFICIAL evaluate_tool_calls.py / evaluate_pass_rate.py score it.

What this is NOT
----------------
It is not a live LiveKit run: there is no real-time audio streaming, no TTS audio and therefore no
official audio-derived latency (user_speech_end_rel / audio_agent_speech_start are not fabricated — they
are omitted). "transcript" is the agent's spoken TEXT (what TTS would say), not an ASR of agent audio.
Report these numbers as "official data + official evaluator, offline replay".

    python3 livekit_agent/fdb_v3_offline_replay.py --data <fdb_v3_data_released> [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FDB_REPO = ROOT / "livekit_agent" / ".fdb_v3_repo" / "v3"

from livekit_agent.adapter import TriageAdapter  # noqa: E402
from livekit_agent.fdb_tools import FDB_TOOLS  # noqa: E402

TEXT_MODE = False  # set by --text
GAP_S = float(os.environ.get("REPLAY_GAP_S", "0.5"))   # pause that ends an ASR segment (LiveKit-like finals)
COMMIT_S = float(os.environ.get("REPLAY_COMMIT_S", "1.6"))  # adapter commit gate, same default as live (C2)
# Importing this module must not change the browser's interactive safety policy.
SETTLE_S = 0.15    # let the agent's queue drain between events


def load_registry():
    # Bundled upstream mocks also work without a separate benchmark clone. APPEND (never insert at 0):
    # the upstream v3/ directory also contains a cascaded_agent.py which must never shadow ours
    # (the bundled mock_apis.py / latency_injector.py are byte-identical copies anyway).
    p = str(FDB_REPO if FDB_REPO.is_dir() else ROOT / "livekit_agent")
    if p not in sys.path:
        sys.path.append(p)
    import mock_apis  # official, unmodified
    return mock_apis.MockAPIRegistry(latency_profile="instant", enable_logging=False)


_ASR = None


def asr_turns(wav: str):
    global _ASR
    from faster_whisper import WhisperModel
    if _ASR is None:
        _ASR = WhisperModel(os.environ.get("ASR_MODEL", "base.en"), device="cpu", compute_type="int8")
    from livekit_agent.speech_providers import bias_terms, whisper_prompt
    prompt = whisper_prompt(bias_terms(FDB_TOOLS)) or None     # same tool-vocabulary biasing as live (C1)
    segs, _ = _ASR.transcribe(wav, beam_size=1, vad_filter=True, temperature=0.0, initial_prompt=prompt)
    turns, cur, last_end = [], [], None
    for s in segs:
        if last_end is not None and s.start - last_end >= GAP_S and cur:
            turns.append({"text": " ".join(x.text.strip() for x in cur), "start": cur[0].start, "end": cur[-1].end})
            cur = []
        cur.append(s)
        last_end = s.end
    if cur:
        turns.append({"text": " ".join(x.text.strip() for x in cur), "start": cur[0].start, "end": cur[-1].end})
    return turns


async def replay(example_dir: Path, registry, provider: str):
    meta = json.loads((example_dir / "metadata.json").read_text())
    t0 = time.time()
    if TEXT_MODE:  # official human transcripts instead of ASR (isolates agent logic from ASR errors)
        turns = [{"text": d["user"], "start": i, "end": i} for i, d in enumerate(meta.get("dialogue") or [])
                 if d.get("user")]
    else:
        turns = asr_turns(str(example_dir / "input.wav"))
    asr_s = time.time() - t0
    calls, spoken = [], []

    async def execute(cid, api, args):
        ts = time.time()
        try:
            res = await asyncio.to_thread(registry.call, api, **args)
        except Exception as e:  # official mocks raise on bad kwargs → structured error
            res = {"status": "error", "error": "invalid_args", "message": f"{type(e).__name__}: {e}"}
        calls.append({"function": api, "args": args, "timestamp_start": round(ts - t0, 3),
                      "timestamp_end": round(time.time() - t0, 3), "call_id": cid})
        if isinstance(res, dict) and "status" not in res:
            res = {"status": "success", **res}
        st = "error" if (isinstance(res, dict) and res.get("status") == "error") else "success"
        await adapter.on_tool_completed(cid, res, status=st)

    async def cancel(cid):
        pass

    async def speak(kind, text):
        spoken.append({"kind": kind, "text": text, "t": round(time.time() - t0, 3)})

    # audio mode replays each ASR segment as a separate final at its real (compressed) time offset
    # through the same commit gate as live, so fragmentation is exercised exactly as in LiveKit (C2)
    commit = 0.0 if TEXT_MODE else COMMIT_S / 8
    adapter = TriageAdapter(tool_executor=execute, tool_canceller=cancel, speak=speak,
                            settle_s=commit, max_settle_s=commit * 1.6)
    await adapter.start(FDB_TOOLS)
    replay_status = "completed"
    try:
        for i, t in enumerate(turns):
            await adapter.on_user_final(t["text"])
            nxt = turns[i + 1]["start"] if i + 1 < len(turns) else t["end"] + 3.0
            gap = max(0.0, nxt - t["end"]) / 8 if not TEXT_MODE else SETTLE_S * 2
            await asyncio.sleep(max(gap, 0.02))
        await adapter.wait_idle(timeout=float(os.environ.get("REPLAY_TIMEOUT_S", "30")))
    except TimeoutError:
        replay_status = "timeout"
    finally:
        await adapter.stop()
    result = {
        "example_id": meta["id"], "provider": provider, "status": replay_status,
        "mode": "offline_replay_official_data",
        "user_turns_asr": turns, "asr_seconds": round(asr_s, 2),
        "actual_tool_calls": [{k: v for k, v in c.items() if k != "call_id"} for c in calls],
        "transcript": " ".join(s["text"] for s in spoken),
        "agent_actions": spoken,
        "expected_tool_calls": meta.get("expected_tool_calls"),
    }
    (example_dir / f"result_{provider}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


async def console(turns=None):
    """Key-free text console using the same adapter and actual bundled mock tools.

    Interactive input does not await tools, so typing a correction interrupts work.
    --say is sequential and drains each request for scriptable smoke tests.
    """
    registry = load_registry()

    async def execute(cid, api, args):
        print(f"TOOL {api} {json.dumps(args)}", flush=True)
        try:
            result = await asyncio.to_thread(registry.call, api, **args)
        except TypeError as exc:
            result = {"status": "error", "error": "invalid_args", "message": str(exc)}
        status = "error" if result.get("status") == "error" else "success"
        await adapter.on_tool_completed(cid, result, status=status)

    async def speak(kind, text):
        print(f"AGENT [{kind}] {text}", flush=True)

    async def cancel(cid):
        print(f"CANCEL {cid}", flush=True)

    adapter = TriageAdapter(tool_executor=execute, tool_canceller=cancel, speak=speak)
    await adapter.start(FDB_TOOLS)
    print("Offline text mode; local mock tools only. No speech APIs or real transactions.")
    try:
        if turns:
            for text in turns:
                print(f"USER {text}", flush=True)
                await adapter.on_user_final(text)
                await adapter.wait_idle()
        else:
            print("Type requests or corrections at any time; /quit exits.")
            while True:
                try:
                    text = await asyncio.to_thread(input, "> ")
                except EOFError:
                    break
                if text.strip() == "/quit":
                    break
                await adapter.on_user_final(text)
            await adapter.wait_idle()
    finally:
        await adapter.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true", help="key-free text console, no dataset needed")
    ap.add_argument("--say", action="append", help="run a text request with bundled tools; repeat for follow-ups")
    ap.add_argument("--data", default=str(ROOT / "livekit_agent" / "fdb_v3_data_released"))
    ap.add_argument("--provider", default="bilio")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--text", action="store_true",
                    help="use the official transcript text instead of Whisper ASR (diagnostic mode)")
    a = ap.parse_args()
    global TEXT_MODE
    TEXT_MODE = a.text
    if a.text and a.provider == "bilio":
        a.provider = "bilio_text"
    # Explicit offline means no hosted planner even if keys are exported in the shell.
    os.environ["TRIAGELINE_LLM_PLANNER"] = "0"
    if a.interactive or a.say:
        asyncio.run(console(a.say))
        return
    if not Path(a.data).is_dir():
        ap.error(f"data directory not found: {a.data}; run ./run_fdb_v3.sh --offline-text first")
    os.environ.setdefault("TRIAGELINE_BENCHMARK_POLICY", "1")
    dirs = sorted(p for p in Path(a.data).iterdir() if (p / "metadata.json").exists())
    if a.only:
        dirs = [d for d in dirs if d.name.startswith(a.only)]
    if a.limit:
        dirs = dirs[: a.limit]
    if not dirs:
        ap.error("no metadata.json examples found in the selected data directory")
    ok = 0
    for i, d in enumerate(dirs, 1):
        r = asyncio.run(replay(d, load_registry(), a.provider))
        exp = [c["function"] for c in r["expected_tool_calls"] or []]
        got = [c["function"] for c in r["actual_tool_calls"]]
        hit = sorted(exp) == sorted(got)
        ok += hit
        print(f"[{i}/{len(dirs)}] {'OK ' if hit else 'MIS'} {d.name[:40]:40s} exp={exp} got={got}", flush=True)
    print(f"\nexact tool-name multiset match: {ok}/{len(dirs)}  (official scores: run evaluate_tool_calls.py)")


if __name__ == "__main__":
    main()
