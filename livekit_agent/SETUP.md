# livekit_agent — BILIO custom LiveKit agent for FDB-v3

> Current setup: see README §B and docs/FREE_API_KEYS.md. The notes below are historical (the agent is no longer the unmodified template; it uses ParticipantAgent, no LLM).

Status: **template copied and verified to import/start its CLI; NOT yet run against a LiveKit room**
(no LiveKit/OpenAI credentials and no outbound access to LiveKit Cloud/OpenAI were available in the
authoring sandbox). Steps below are the exact ones to finish that check.

## Provenance
Files copied byte-for-byte from `github.com/DanielLin94144/Full-Duplex-Bench` (`v3/`), no edits:
`cascaded_agent.py` (agent), `mock_apis.py` + `latency_injector.py` (imported by the agent),
`livekit_inference.py` (FDB's headless test client that streams a WAV into a room).
Pipeline: Silero VAD -> hosted STT -> rule-based ParticipantAgent (no LLM, 12 mock tools) -> hosted TTS.

## Accounts / keys
1. LiveKit Cloud account + project (free tier OK): https://cloud.livekit.io
2. Speech keys (Groq/Deepgram free, or OpenAI); OpenAI gpt-4o only for the official judge.

## Environment variables (names only; see `.env.example`)
| Var | Used by |
|---|---|
| `LIVEKIT_URL` (wss://<project>.livekit.cloud) | agent + test client |
| `LIVEKIT_API_KEY` | agent + test client |
| `LIVEKIT_API_SECRET` | agent + test client |
| `OPENAI_API_KEY` | agent (STT + LLM + TTS) |

## LiveKit Cloud configuration
- Create a project; copy the WebSocket URL and generate an API key/secret pair.
- No dispatch rules needed: the agent registers without an `agent_name`, so it is auto-dispatched
  to any room the test client creates (default room `test-room`).

## Install (tested versions: Python 3.12, livekit-agents 1.8.3; FDB README targets py3.10 / ~=1.3)
```bash
cd livekit_agent
python -m venv .venv && source .venv/bin/activate
pip install "livekit-agents[openai]~=1.3" livekit-plugins-silero \
            livekit livekit-api python-dotenv numpy
cp .env.example .env.local   # then fill in values
```
Note: `pip install "livekit[crypto]"` warns that the extra doesn't exist; the plain package is fine.
`pip install "livekit-agents[openai]~=1.3"` resolved to 1.8.3 here and the template imported fine.

## Run
Terminal 1 (agent; run from `livekit_agent/` so `.env.local` and `mock_apis` resolve):
```bash
python cascaded_agent.py start
```
Terminal 2 (send a test utterance; any speech WAV, FDB data is 48 kHz):
```bash
python livekit_inference.py -i input.wav -o response.wav --room test-room
```
Expected: Terminal 1 prints `CASCADED AGENT JOINING ROOM`, `STT: '<your words>'`, tool-call logs,
and `response.wav` contains the agent's spoken reply. Alternative local check with mic/speaker:
`python cascaded_agent.py console`.

## Known template quirks (left unmodified)
- Writes logs to `/tmp/agent_heartbeat.log` and `/tmp/agent_tool_calls.log`.
- `--latency <profile>` custom arg is parsed before the LiveKit CLI.
- FDB benchmark audio (Google Drive, `fdb_v3_data_released/`) is only needed for scoring, not this smoke test.
