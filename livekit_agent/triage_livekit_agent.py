#!/usr/bin/env python3
"""
Triage Line voice agent -- roadside/incident-triage, running on the SAME
LiveKit agent shell as the FDB-v3 cascaded_agent.py (this is a second call
flow inside the shared shell, not a second LiveKit agent process).

Pipeline (no LLM anywhere):
  Caller Audio -> Silero VAD -> hosted STT (TRIAGELINE_STT_PROVIDER: openai | groq | deepgram)
      -> TriageCallSession (legacy dialogue -> deliberation -> commit state
         machine, via livekit_agent/triage_brain.py)
      -> hosted TTS (TRIAGELINE_TTS_PROVIDER: openai | deepgram) -> Agent Audio

STT: this agent passes stt=None to TriageCallSession on purpose -- LiveKit's AgentSession owns STT and
delivers transcripts through `user_input_transcribed` (audit §5 item 3).

Status: built against livekit-agents 1.8.3 (AgentServer, `@server.rtc_session()`,
`user_input_transcribed`, `agent_state_changed`) and imports cleanly; the confirmation-safety logic it
drives is covered offline by adapter_tests/test_4 and test_8. A run in a live LiveKit room needs your own
LiveKit Cloud project (free Build plan) -- see docs/FREE_API_KEYS.md. Barge-in: a partial transcript that
arrives while the agent is speaking is treated as caller speech onset (fast path); the final transcript
is the signal of record.

Usage (without TRIAGELINE_AGENT_NAME both workers are auto-dispatched, so run ONE at a time; with
distinct names, both can share a LiveKit project and the gateway token decides which one joins):
    python livekit_agent/triage_livekit_agent.py dev      # connect to your LiveKit project
    python livekit_agent/triage_livekit_agent.py console  # local mic/speaker smoke test

Environment (livekit_agent/.env.local, same file cascaded_agent.py reads):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET + the chosen speech provider key(s)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobProcess
# Silero registers its LiveKit plugin at import time and requires the process main thread.
# Keep this eager import here rather than first importing it inside setup_fnc/job code.
from livekit.plugins import silero as _silero  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):   # repo root (livekit_agent.*) + this dir (triage_brain)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from triage_brain import TriageCallSession  # noqa: E402

env_path = os.path.join(os.path.dirname(__file__), ".env.local")
load_dotenv(env_path)

log = logging.getLogger("triage_livekit_agent")


# --------------------------------------------------------------------------- provider adapters
# legacy/core's DialogueEngine needs concrete AudioIO/TTSProvider objects
# (see providers/interfaces.py). These adapt the live LiveKit AgentSession's
# playback into that interface, the same way providers/mock/mock_*.py adapt
# a scripted test double into it -- DialogueEngine itself is unmodified.

_LEGACY_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy")
if _LEGACY_ROOT not in sys.path:
    sys.path.insert(0, _LEGACY_ROOT)
from providers.interfaces import AudioIO, TTSProvider  # noqa: E402


class LiveKitTTSAdapter(TTSProvider):
    """Routes DialogueEngine's speech into the real AgentSession's TTS via
    session.say(), instead of a language model deciding what to say.
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._speaking = False
        self._interrupted = False
        self._handle = None

    async def start_speaking(self, text: str) -> None:
        self._speaking = True
        self._interrupted = False
        # allow_interruptions defaults on in AgentSession; caller barge-in
        # will stop this speech handle the same way it stops a normal
        # LLM-generated reply.
        self._handle = self._session.say(text)

    async def stop(self) -> None:
        if self._handle is not None:
            self._interrupted = self._speaking
            try:
                self._handle.interrupt()
            except Exception:
                log.warning("could not interrupt in-flight TTS handle", exc_info=True)
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    def progress(self) -> float:
        # AgentSession's SpeechHandle does not expose sub-utterance progress
        # in a stable, version-independent way; 0.0/1.0 is an honest
        # placeholder (same simplification legacy/providers/mock/mock_tts.py
        # makes explicit via its own word-count estimate) -- only used by
        # DialogueEngine as a barge-in-position metric, not for correctness.
        return 0.0 if self._speaking else 1.0

    def was_interrupted(self) -> bool:
        return self._interrupted


class LiveKitAudioIOAdapter(AudioIO):
    """Caller-speech / agent-playback signal, fed by AgentSession callbacks
    (registered in entrypoint() below) rather than polled from hardware.
    """

    def __init__(self) -> None:
        self._caller_speaking = False
        self._agent_playing = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._caller_speaking = False
        self._agent_playing = False

    def is_caller_speaking(self) -> bool:
        return self._caller_speaking

    def is_agent_playing(self) -> bool:
        return self._agent_playing

    async def stop_agent_playback(self) -> None:
        self._agent_playing = False

    # driven by session event callbacks, not by DialogueEngine
    def set_caller_speaking(self, value: bool) -> None:
        self._caller_speaking = value

    def set_agent_playing(self, value: bool) -> None:
        self._agent_playing = value


def build_triage_pipeline(vad=None):
    """Same VAD/STT/TTS construction as cascaded_agent.py (shared speech_providers module), so this is
    recognizably the same shell, just with the triage brain instead of the FDB tool agent. No LLM."""
    from livekit_agent.speech_providers import build_pipeline
    vad, stt, tts, _cfg = build_pipeline(vad)
    return vad, stt, tts


class TriageVoiceAgent(Agent):
    def __init__(self) -> None:
        # No LLM-authored instructions: this agent's replies come entirely
        # from TriageCallSession/TriageBrainLLM (deliberation + commit
        # state machine), never from free-form generation.
        super().__init__(instructions="")


def prewarm(proc: JobProcess):
    """Load Silero in the worker process main thread before room jobs run.

    LiveKit executes room entrypoints on job threads, where importing the Silero plugin
    for the first time raises "Plugins must be registered on the main thread".
    """
    from livekit_agent.speech_providers import load_vad
    proc.userdata["vad"] = load_vad()


server = AgentServer(setup_fnc=prewarm)
# Explicit dispatch: set TRIAGELINE_AGENT_NAME (e.g. bilio-triage) and the same value on the
# gateway so its tokens route calls here; empty keeps automatic dispatch (demo only, run one worker).
AGENT_NAME = os.environ.get("TRIAGELINE_AGENT_NAME", "").strip()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: agents.JobContext):
    call_id = ctx.room.name
    print(f"!!! TRIAGE LINE AGENT JOINING ROOM: {call_id} !!!")

    vad, stt, tts = build_triage_pipeline(getattr(ctx.proc, "userdata", {}).get("vad"))

    # AgentSession still owns the real audio I/O (mic in, speaker out) and
    # VAD; what it does NOT own is deciding what to say -- that is
    # delegated to TriageCallSession via LiveKitTTSAdapter.say(), driven by
    # STT transcripts below, not by session.generate_reply()/an LLM.
    session = AgentSession(vad=vad, stt=stt, tts=tts,
                           turn_handling={"endpointing": {"min_delay": 0.5, "max_delay": 5.0}})

    audio_io = LiveKitAudioIOAdapter()
    tts_adapter = LiveKitTTSAdapter(session)
    triage = TriageCallSession(call_id=call_id, stt=None, tts=tts_adapter, audio_io=audio_io)
    final_segments: list[str] = []
    flush_handle: asyncio.TimerHandle | None = None

    async def flush_final_transcript() -> None:
        """Coalesce Deepgram final segments; a final segment is not necessarily a whole turn."""
        try:
            text = " ".join(s.strip() for s in final_segments if s.strip()).strip()
            final_segments.clear()
            if text:
                log.info("processing caller turn: %s", text)
                await triage.on_final_transcript(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("failed to process caller transcript for call %s", call_id)

    def schedule_transcript_flush(delay: float) -> None:
        nonlocal flush_handle
        if flush_handle is not None:
            flush_handle.cancel()

        def launch_flush() -> None:
            nonlocal flush_handle
            flush_handle = None
            asyncio.create_task(flush_final_transcript())

        flush_handle = asyncio.get_running_loop().call_later(delay, launch_flush)

    @session.on("user_input_transcribed")
    def _on_transcript(msg: agents.voice.UserInputTranscribedEvent):
        print(f"  📝 STT: '{msg.transcript}' (final={msg.is_final})")
        if msg.is_final:
            final_segments.append(msg.transcript)
            # Prefer the user-state transition as the turn boundary; keep a short
            # debounce fallback in case this LiveKit version omits that transition.
            schedule_transcript_flush(1.5)
        else:
            audio_io.set_caller_speaking(True)
            asyncio.create_task(triage.on_partial_transcript(msg.transcript))
            if audio_io.is_agent_playing():
                # Fast-path barge-in detection: a partial transcript arriving
                # while the agent is mid-utterance is caller speech onset.
                asyncio.create_task(triage.check_for_barge_in())

    @session.on("user_state_changed")
    def _on_user_state(ev):
        speaking = ev.new_state == "speaking"
        audio_io.set_caller_speaking(speaking)
        if speaking and audio_io.is_agent_playing():
            asyncio.create_task(triage.check_for_barge_in())
        elif not speaking and final_segments:
            schedule_transcript_flush(0.35)

    @session.on("agent_state_changed")
    def _on_agent_state(ev: agents.voice.AgentStateChangedEvent):
        audio_io.set_agent_playing(ev.new_state == "speaking")
        tts_adapter._speaking = ev.new_state == "speaking"

    # --- fix for the disconnect issue named in the extension brief ---
    # No action may be left non-terminal after the call ends. This must run
    # on every teardown path (explicit disconnect or job shutdown), not just
    # a clean hangup, so it is registered against both.
    torn_down = False

    async def _teardown(*_args, **_kwargs) -> None:
        nonlocal torn_down
        if torn_down:
            return
        torn_down = True
        resolved = await triage.teardown(reason="caller_disconnected")
        if resolved:
            log.info(
                "force_resolve_pending on teardown resolved %d action(s) for call %s: %s",
                len(resolved), call_id, [(r.action_id, r.current_state.value) for r in resolved],
            )

    ctx.room.on("disconnected", lambda *a, **k: asyncio.create_task(_teardown()))
    ctx.add_shutdown_callback(_teardown)

    await session.start(room=ctx.room, agent=TriageVoiceAgent())
    asyncio.create_task(tts_adapter.start_speaking("Welcome to BILIO Triage! How can I help you?"))
    from livekit_agent.speech_providers import describe
    print(f"!!! TRIAGE LINE AGENT STARTED: {describe().replace('rule-based agent', 'triage brain')} !!!")


if __name__ == "__main__":
    from livekit_agent.speech_providers import preflight
    preflight(sys.argv[1] if len(sys.argv) > 1 else "")
    agents.cli.run_app(server)
