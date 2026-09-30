# Free API keys: running BILIO at no cost

One **Google AI Studio key** is enough for everything except the transport: Gemini does the speech-to-text,
the text-to-speech and the optional LLM planner. The free LLM providers below act only as automatic
failover for the planner. Groq and Deepgram are optional alternatives, not requirements.

| Need | Free option (checked 2026-09) | Env var(s) |
|---|---|---|
| Realtime transport | **LiveKit Cloud "Build" plan**: no card; 1,000 agent-session minutes and 5,000 WebRTC minutes a month | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| STT + TTS + planner | **Google AI Studio (Gemini)** free tier. Defaults: `gemini-3.5-flash-lite` (STT and planner) and `gemini-3.8-flash-lite-tts` (TTS) | `GEMINI_API_KEY` |
| Planner failover (optional) | **Cerebras**, **OpenRouter** (`:free` models) and **Mistral** (experiment tier), all OpenAI-compatible | `CEREBRAS_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`; order with `TRIAGELINE_LLM_CHAIN=gemini,cerebras,openrouter,mistral` |
| Streaming speech (optional, lower latency) | **Deepgram**: $200 signup credit (`nova-3` STT, Aura-2 TTS) | `DEEPGRAM_API_KEY` |
| Official LLM judge (evaluation only) | The official evaluators hard-code **`gpt-4o`**. No free tier exists; a full 100-example pass costs well under $1 | `OPENAI_API_KEY` |

Free-tier quotas and model names change. Run `python3 scripts/check_providers.py` to confirm each configured key with one tiny request.

## 1. LiveKit Cloud
1. Sign up at <https://cloud.livekit.io> (GitHub or Google login, no card), then create a project.
2. Go to **Settings → API Keys → Create key**. Copy the WebSocket URL (`wss://<project>.livekit.cloud`), the key and the secret.
3. Optional: set `TRIAGELINE_AGENT_NAME=bilio` on the worker **and** the gateway. The worker then joins only the rooms whose tokens request it (explicit dispatch), which lets several workers share one project.

## 2. Gemini (Google AI Studio)
1. Open <https://aistudio.google.com/apikey> → **Create API key**.
2. Set `GEMINI_API_KEY=...`. With `TRIAGELINE_*_PROVIDER=auto` (the default), Gemini is selected for STT, TTS and the planner.
3. On an older project without the 3.x models, set `TRIAGELINE_STT_MODEL=gemini-2.5-flash` (and the matching TTS model).

## 3. Failover providers (optional)
- Cerebras: <https://cloud.cerebras.ai> → API keys.
- OpenRouter: <https://openrouter.ai/keys>. Free models end in `:free` and allow about 50 requests a day without credit.
- Mistral: <https://console.mistral.ai> → API keys (free experiment plan).

The planner retries once on a 429, a 5xx or a timeout, then moves to the next provider. Bad keys (401/400) are skipped at once.

## 4. Put the keys in place
```bash
cp .env.example .env                  # gitignored. The worker also reads livekit_agent/.env.local
LIVEKIT_URL=wss://<project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
GEMINI_API_KEY=...
# CEREBRAS_API_KEY=...  OPENROUTER_API_KEY=...  MISTRAL_API_KEY=...
# OPENAI_API_KEY=sk-...                # only for the official gpt-4o judge
```

## 5. Run
```bash
./run_fdb_v3.sh --offline-text                          # no keys: text replay (diagnostic only)
PYTHON=python3.12 ./run_fdb_v3.sh --limit 5 --require-judge   # judged live smoke run
PYTHON=python3.12 ./run_fdb_v3.sh --require-judge             # full scored run
python -m ui                                            # PWA + /rtc.html voice (same keys)
python livekit_agent/cascaded_agent.py dev              # the voice worker
```
