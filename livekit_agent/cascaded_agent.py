#!/usr/bin/env python3
"""
BILIO cascaded voice agent for the official FDB-v3 benchmark.

MODEL / PROVIDER DECLARATION (guide: "clear declaration of the model provider or custom agent")
------------------------------------------------------------------------------------------------
This is a CUSTOM LiveKit agent (the exact providers/models are printed at start-up by describe()):

  User audio -> Silero VAD (local)            [livekit-plugins-silero]
             -> hosted STT                    [Gemini gemini-3.5-flash-lite | Deepgram nova-3 |
                                               OpenAI whisper-1 | Groq — TRIAGELINE_STT_PROVIDER, auto]
             -> TriageAdapter  (livekit_agent/adapter.py: non-blocking tool tasks, utterance
                                settling, barge-in)
                -> ParticipantAgent (agent/agent.py + agent/nlu.py: schema-driven tool selection +
                   argument extraction, epoch-guarded interruption handling, cancellation,
                   duplicate-action ledger) + OPTIONAL schema-validated LLM planner through the
                   failover chain in agent/providers.py (Gemini -> Cerebras -> OpenRouter -> Mistral)
             -> FDB-v3 mock tools (official mock_apis.py, unmodified)
             -> hosted TTS                    [Gemini gemini-3.8-flash-lite-tts (streamed) | Deepgram aura-2 |
                                               OpenAI tts-1 — TRIAGELINE_TTS_PROVIDER, auto]

The upstream FDB-v3 template (v3/cascaded_agent.py) lets gpt-4o call tools directly. Here every
LLM proposal is validated against the tool schema and every state-changing call goes through one
epoch-tracked, deduplicated path (see docs/ARCHITECTURE.md).

MODES (TRIAGELINE_MODE):
  benchmark (default)  FDB-v3 template behaviour: never asks clarifying questions, calls with the
                       known arguments, no confirmation gate. Auto-dispatched into every room.
  assistant            phone assistant: asks for missing details, confirms LLM-proposed side effects,
                       conversational replies. Set TRIAGELINE_AGENT_NAME so the gateway's tokens
                       dispatch only this worker (explicit dispatch).

Telemetry contract with the official runner (v3/run_tool_benchmark.py):
  /tmp/agent_heartbeat.log   "!!! CASCADED AGENT JOINING ROOM ..." + "LATENCY_TRACK_JSON: {...}"
  /tmp/agent_tool_calls.log  one JSON line per tool call {"room", "call": {function,args,ts}}

Usage:
    python cascaded_agent.py start            # production worker (used by run_fdb_v3.sh)
    python cascaded_agent.py dev | console    # local development
    python cascaded_agent.py start --latency normal

Environment (.env.local next to this file, or exported):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    + the key of the chosen speech provider(s): OPENAI_API_KEY | GROQ_API_KEY | DEEPGRAM_API_KEY
    optional: TRIAGELINE_STT_PROVIDER / TRIAGELINE_TTS_PROVIDER (default openai),
              TRIAGELINE_STT_MODEL / TRIAGELINE_TTS_MODEL / TRIAGELINE_TTS_VOICE,
              TRIAGELINE_SETTLE_S (default 1.0 benchmark, 0.9 assistant), TRIAGELINE_MAX_SETTLE_S (default 2.0),
              TRIAGELINE_BACKCHANNEL (default 1: one "Mm-hm." per substantive turn, not counted as the response)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobProcess

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_REPO_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from livekit_agent.adapter import TriageAdapter, attach_livekit_session  # noqa: E402
from livekit_agent.fdb_tools import FDB_TOOLS  # noqa: E402

# Parse custom CLI args before LiveKit CLI
LATENCY_PROFILE = "instant"
if "--latency" in sys.argv:
    idx = sys.argv.index("--latency")
    if idx + 1 < len(sys.argv):
        LATENCY_PROFILE = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

load_dotenv(os.path.join(_HERE, ".env.local"))

# Construct a fresh official backend per room: no state may cross conversations.
from mock_apis import MockAPIRegistry

log = logging.getLogger("bilio.cascaded_agent")

MODE = os.environ.get("TRIAGELINE_MODE", "benchmark").strip().lower()
if MODE not in ("benchmark", "assistant"):
    raise SystemExit("TRIAGELINE_MODE must be benchmark or assistant")
# The official runner reads these fixed paths; overridable for containers / assistant deployments.
HEARTBEAT = os.environ.get("TRIAGELINE_HEARTBEAT_LOG", "/tmp/agent_heartbeat.log")
TOOL_LOG = os.environ.get("TRIAGELINE_TOOL_LOG", "/tmp/agent_tool_calls.log")
# Commit gate (C2). VAD endpointing (0.5 s) + the semantic turn detector already decide end-of-turn;
# this gate only merges endpointer-split fragments, so it is kept short. Turns that still look
# unfinished (dangling connective, a ranked tool missing a required argument) wait MAX_SETTLE_S.
_default_settle = "1.0" if MODE == "benchmark" else "0.9"
SETTLE_S = float(os.environ.get("TRIAGELINE_SETTLE_S", _default_settle))
MAX_SETTLE_S = float(os.environ.get("TRIAGELINE_MAX_SETTLE_S", "2.0"))
# FDB-v3 template never asks clarifying questions; the scorer checks expected args only (C4).
# Assistant mode asks and confirms instead.
os.environ.setdefault("TRIAGELINE_BENCHMARK_POLICY", "1" if MODE == "benchmark" else "0")
BACKCHANNEL = os.environ.get("TRIAGELINE_BACKCHANNEL", "1") == "1"
# Only backchannel on turns long enough to need processing time (guide penalises excessive fillers).
BACKCHANNEL_MIN_WORDS = int(os.environ.get("TRIAGELINE_BACKCHANNEL_MIN_WORDS", "6"))
AGENT_NAME = os.environ.get("TRIAGELINE_AGENT_NAME", "").strip()


def should_backchannel(transcript: str, busy: bool, already: bool) -> bool:
    """One short acknowledgement per substantive user TURN (``already`` = this turn was already
    acknowledged), only while nothing else is speaking. Short turns get none (no excessive fillers)."""
    return BACKCHANNEL and not busy and not already and len((transcript or "").split()) >= BACKCHANNEL_MIN_WORDS


def _append(path: str, *lines: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except OSError:
        pass


async def append_async(path: str, *lines: str) -> None:
    """File telemetry off the event loop (ruff ASYNC230 / audit §7)."""
    await asyncio.to_thread(_append, path, *lines)


# ---------------------------------------------------------------------------
# Latency tracker — same record shape as the upstream template, but tracks every
# tool call separately (chained / concurrent calls no longer overwrite each other).
# ---------------------------------------------------------------------------
class LatencyTracker:
    def __init__(self):
        self.user_done_at = 0.0
        self.agent_start_at = 0.0          # first agent speech of ANY kind (filler counts)
        self.calls: list[dict] = []        # [{tool, start, end}]
        self.query_received = False
        self.logged = False

    def tool_started(self, tool: str) -> dict:
        rec = {"tool": tool, "start": time.time(), "end": 0.0}
        self.calls.append(rec)
        return rec

    def breakdown(self, room_name: str) -> tuple[str, str] | None:
        if not self.user_done_at or not self.agent_start_at:
            return None
        first = self.calls[0] if self.calls else None
        tool_start = first["start"] if first else 0.0
        tool_end = max((c["end"] for c in self.calls if c["end"]), default=0.0)
        reasoning = max(0.0, tool_start - self.user_done_at) if tool_start else 0.0
        execution = max(0.0, tool_end - tool_start) if tool_start and tool_end else 0.0
        # the spoken acknowledgement can start BEFORE the tool returns (fast path): synthesis is then
        # 0 by definition (speech was not waiting on the tool), never a negative number
        synthesis = max(0.0, self.agent_start_at - (tool_end or self.user_done_at))
        total = max(0.0, self.agent_start_at - self.user_done_at)
        tool_name = ",".join(c["tool"] for c in self.calls) or "none"
        report = (f"\nLATENCY BREAKDOWN ({tool_name}) for room {room_name}:\n"
                  f"  - Reasoning (ASR -> tool decision): {reasoning:.2f}s\n"
                  + (f"  - Tool Execution (API):    {execution:.2f}s\n" if execution else "")
                  + f"  - Synthesis (Tool -> Spoken): {synthesis:.2f}s\n"
                  f"  - TOTAL SEARCH LATENCY:      {total:.2f}s\n")
        metrics = {"room": room_name, "tool": tool_name, "reasoning": round(reasoning, 3),
                   "execution": round(execution, 3), "synthesis": round(synthesis, 3),
                   "total": round(total, 3), "agent_start_at": self.agent_start_at,
                   "tool_calls": [{"tool": c["tool"], "start": c["start"], "end": c["end"]} for c in self.calls]}
        return report, f"LATENCY_TRACK_JSON: {json.dumps(metrics)}"


class TurnTimeline:
    """Per-turn and barge-in timing (every turn, not only the first; the official record is unchanged).

    Writes one ``TURN_LATENCY_JSON`` line per event to the heartbeat log:
      * ``turn``      final transcript -> first agent audio of ANY kind (``first_audio_s``, a backchannel
                      counts) and -> first SUBSTANTIVE audio (``substantive_s`` = ``response_s``;
                      the "Mm-hm." backchannel never counts as the response)
      * ``barge_in``  user speech onset while the agent speaks -> agent audio stopped (``stop_s``)
    """
    def __init__(self, room: str):
        self.room = room
        self.turn_at = 0.0
        self.first_audio_s: float | None = None
        self.turn_text = ""
        self.agent_speaking = False
        self.barge_at = 0.0
        self.records: list[dict] = []

    def final_transcript(self, text: str) -> None:
        self.turn_at, self.turn_text, self.first_audio_s = time.time(), (text or "")[:120], None

    def user_started(self) -> None:
        if self.agent_speaking and not self.barge_at:
            self.barge_at = time.time()

    def agent_state(self, new: str, backchannel: bool = False) -> dict | None:
        now, rec = time.time(), None
        if new == "speaking":
            self.agent_speaking = True
            if self.turn_at and self.first_audio_s is None:
                self.first_audio_s = round(now - self.turn_at, 3)
            if self.turn_at and not backchannel:
                sub = round(now - self.turn_at, 3)
                rec = {"type": "turn", "room": self.room, "first_audio_s": self.first_audio_s,
                       "substantive_s": sub, "response_s": sub, "text": self.turn_text, "at": now}
                self.turn_at, self.first_audio_s = 0.0, None
        else:
            if self.agent_speaking and self.barge_at:
                rec = {"type": "barge_in", "room": self.room, "stop_s": round(now - self.barge_at, 3), "at": now}
            self.agent_speaking, self.barge_at = False, 0.0
        if rec:
            self.records.append(rec)
            del self.records[:-200]
        return rec


class CascadedVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="")


def build_turn_handling() -> dict:
    """livekit-agents 1.8 TurnHandlingOptions (B16: replaces deprecated min/max_endpointing_delay).
    Uses the semantic end-of-turn model when livekit-plugins-turn-detector is installed (C2)."""
    th: dict = {"endpointing": {"min_delay": 0.5, "max_delay": 5.0}}
    if os.environ.get("TRIAGELINE_TURN_DETECTOR", "1") == "1":
        try:
            from livekit_agent.speech_providers import _allow_plugin_registration_in_subprocess
            _allow_plugin_registration_in_subprocess()
            from livekit.plugins.turn_detector.multilingual import MultilingualModel
            th["turn_detection"] = MultilingualModel()
        except Exception as e:  # noqa: BLE001 - plugin optional; VAD endpointing + commit gate still apply
            log.info("semantic turn detector unavailable (%s); using VAD endpointing", e)
    return th


def build_cascaded_pipeline(vad=None):
    """VAD + STT + TTS (provider chosen by env, see speech_providers.py). No LLM here."""
    from livekit_agent.speech_providers import build_pipeline
    vad, stt, tts, _cfg = build_pipeline(vad)
    return vad, stt, tts


def prewarm(proc: JobProcess):
    """Load Silero once per worker process, not once per room (audit B-11)."""
    from livekit_agent.speech_providers import load_vad
    proc.userdata["vad"] = load_vad()


server = AgentServer(setup_fnc=prewarm)


# agent_name='' keeps LiveKit's automatic dispatch (required by the official benchmark runner);
# a name switches to explicit dispatch so only tokens carrying RoomConfiguration(agents=[name]) get it.
@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: agents.JobContext):
    registry = MockAPIRegistry(latency_profile=LATENCY_PROFILE)
    room_name = ctx.room.name
    # no state crosses conversations: provider cooldowns / last-status from a previous room are reset
    try:
        from agent import providers as _providers
        _providers.reset_state()
    except Exception:  # noqa: BLE001 - planner optional
        pass
    await append_async(HEARTBEAT, f"!!! CASCADED AGENT JOINING ROOM: {room_name} at {time.ctime()} !!!")
    print(f"!!! CASCADED AGENT JOINING ROOM: {room_name} !!!")

    vad, stt, tts = build_cascaded_pipeline(getattr(ctx.proc, "userdata", {}).get("vad"))
    tracker = LatencyTracker()
    timeline = TurnTimeline(room_name)
    session = AgentSession(vad=vad, stt=stt, tts=tts, turn_handling=build_turn_handling())

    cancelled: set[str] = set()

    async def tool_executor(call_id: str, api_name: str, args: dict) -> None:
        """The ONLY place FDB tools are invoked; runs as its own task (adapter B-05)."""
        rec = tracker.tool_started(api_name)
        if registry is None:
            result = {"status": "error", "error": "unavailable", "message": "mock_apis registry unavailable"}
        else:
            try:
                result = await asyncio.to_thread(registry.call, api_name, **args)
            except TypeError as e:          # official mocks raise on missing/extra kwargs
                result = {"status": "error", "error": "invalid_args", "message": str(e)}
            except Exception as e:  # noqa: BLE001 - never let one bad tool call kill the session
                result = {"status": "error", "error": "error", "message": str(e)}
        rec["end"] = time.time()
        await append_async(TOOL_LOG, json.dumps({"room": room_name, "call": {
            "function": api_name, "args": args, "timestamp_start": rec["start"], "timestamp_end": rec["end"]}}))
        if call_id in cancelled:
            cancelled.discard(call_id)
            log.info("late result for cancelled call_id=%s (%s) -> ledger reconciliation", call_id, api_name)
        status = "error" if isinstance(result, dict) and result.get("status") == "error" else "ok"
        await adapter.on_tool_completed(call_id, result, status=status)

    async def tool_canceller(call_id: str) -> None:
        cancelled.add(call_id)

    async def speak(kind: str, text: str) -> None:
        if text:
            session.say(text, allow_interruptions=True)

    async def interrupt_speech() -> None:
        try:
            session.interrupt()
        except RuntimeError:  # nothing playing / session not running yet
            pass

    adapter = TriageAdapter(tool_executor=tool_executor, tool_canceller=tool_canceller, speak=speak,
                            settle_s=SETTLE_S, max_settle_s=MAX_SETTLE_S, interrupt_speech=interrupt_speech)
    await adapter.start(FDB_TOOLS)

    # per-turn backchannel state: one "Mm-hm." per substantive user turn, never counted as the response
    turn_state = {"acknowledged": False, "backchannel_pending": False}

    def on_transcript(msg) -> None:
        """Telemetry + fast-path acknowledgement; runs inside the adapter's single transcript handler."""
        logging.info("STT TRANSCRIPT: '%s' (is_final=%s)", msg.transcript, msg.is_final)
        if not msg.is_final or not (msg.transcript or "").strip():
            return
        timeline.final_transcript(msg.transcript)
        if not tracker.query_received:
            tracker.user_done_at = time.time()    # same anchor as the upstream template
            tracker.query_received = True
        if should_backchannel(msg.transcript, adapter.busy(), turn_state["acknowledged"]):
            turn_state["acknowledged"] = True
            # immediate acknowledgement while the utterance settles (fast path). It is NOT counted
            # as the first response (B17): latency is stamped on the first substantive line.
            turn_state["backchannel_pending"] = True
            session.say("Mm-hm.", allow_interruptions=True, add_to_chat_ctx=False)

    def on_user_state(ev) -> None:
        if getattr(ev, "new_state", None) == "speaking":
            timeline.user_started()
            turn_state["acknowledged"] = False    # a new user turn may be acknowledged again

    attach_livekit_session(session, adapter, room_name=room_name,
                           on_transcript=on_transcript, on_user_state=on_user_state)

    @session.on("agent_state_changed")
    def on_agent_state(ev: agents.voice.AgentStateChangedEvent):
        is_bc = ev.new_state == "speaking" and turn_state["backchannel_pending"]
        if is_bc:
            turn_state["backchannel_pending"] = False     # the "Mm-hm." itself: not a substantive reply
        rec = timeline.agent_state(ev.new_state, backchannel=is_bc)
        if rec:
            asyncio.create_task(append_async(HEARTBEAT, "TURN_LATENCY_JSON: " + json.dumps(rec)))
        # stamp the moment audio actually starts, not when speech was queued (audit B-07)
        if ev.new_state == "speaking" and not is_bc and tracker.query_received and not tracker.agent_start_at:
            tracker.agent_start_at = time.time()

    async def _flush_latency():
        out = tracker.breakdown(room_name)
        if out and not tracker.logged:
            tracker.logged = True
            print(out[0])
            await append_async(HEARTBEAT, out[0], out[1])

    async def _latency_watch():
        # write the record once the first tool chain has settled (or on teardown)
        while True:
            await asyncio.sleep(1.0)
            if tracker.agent_start_at and not adapter.busy() and not tracker.logged:
                await _flush_latency()

    watch = asyncio.create_task(_latency_watch())

    torn_down = False

    async def _teardown(*_args, **_kwargs) -> None:
        # registered on both room "disconnected" and shutdown: run once (B8). Buffered text is
        # dropped, never routed, so no tool call can be logged after the room is gone.
        nonlocal torn_down
        if torn_down:
            return
        torn_down = True
        watch.cancel()
        adapter.close()
        await _flush_latency()
        await adapter.stop()

    ctx.room.on("disconnected", lambda *a, **k: asyncio.create_task(_teardown()))
    ctx.add_shutdown_callback(_teardown)

    await session.start(room=ctx.room, agent=CascadedVoiceAgent())
    from livekit_agent.speech_providers import describe
    banner = (f"!!! TRIAGELINE CASCADED AGENT STARTED [{MODE}{', agent_name=' + AGENT_NAME if AGENT_NAME else ''}; "
              f"settle={SETTLE_S}s/{MAX_SETTLE_S}s; llm_mode={os.environ.get('TRIAGELINE_LLM_MODE', 'fallback')}]: "
              f"{describe()} !!!")
    print(banner)
    await append_async(HEARTBEAT, banner)


if __name__ == "__main__":
    from livekit_agent.speech_providers import preflight
    preflight(sys.argv[1] if len(sys.argv) > 1 else "")
    agents.cli.run_app(server)
