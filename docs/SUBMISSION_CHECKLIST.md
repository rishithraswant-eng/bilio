# Submission checklist (source of truth — supersedes everything in docs/archive/ and legacy/docs/*)

## Done in code (verified in this repo)
- Single ASGI server `ui/api.py` (`python -m ui` / uvicorn); `ui/server.py` is a thin launcher.
- `run_fdb_v3.sh` fails on runner error, missing/errored/stale results, and `--require-judge` without judge; records effective config (`tests/shell/test_run_fdb_v3_failures.sh`).
- LLM provider failover chain (Gemini → Cerebras → OpenRouter → Mistral), schema validation, confirmation gate for state changes in assistant mode.
- Read-only results grounded on call-time context (`tests/test_grounding.py`); same-turn retraction withdraws the request.
- Worker: `TRIAGELINE_MODE=benchmark|assistant`, explicit dispatch via `TRIAGELINE_AGENT_NAME`, gated backchannel, per-turn and barge-in timing (`TURN_LATENCY_JSON`), triage never claims real dispatch.
- Voice client: fresh room per call, mic pause/resume on backgrounding (hang-up after 3 min), robust agent detection, wake lock; `npm test` 10/10.
- Integrity: `scripts/integrity_audit.py` PASS; independent held-out set `scenarios_heldout/` 29/30 rules-only.
- Packaging: `Dockerfile` (api/worker targets), `docker-compose.yml`, root `.env.example`; CI workflow template at `docs/ci.github-workflow.yml` (copy to `.github/workflows/ci.yml` yourself — the bot token cannot push workflow files).
- UI redesigned (warm editorial theme, rules instead of cards, no blue) across `/live.html`, `/rtc.html` and `/console`.
- Docs: Gemini-first `docs/FREE_API_KEYS.md`, `docs/MOBILE_INTEGRATION.md` (token flow + Flutter snippet), README quick-start/production corrected.
- Tests: 224 Python + 10 JS passing (Python 3.13 sandbox with livekit-agents 1.8.3 installed).
- 2026-09-29 readiness pass: settle timer re-armed on speech onset (no stranded requests); trailing budget clauses merge
  (no duplicate search); state-modifying calls never sent with a required arg missing; chained steps bind earlier
  results and committed filters; rules-first LLM mode in benchmark, multi-call plans executed in order; Deepgram-first
  speech in benchmark mode; per-turn backchannel not counted as the response; un-dispatched state changes really
  cancelled; `run_fdb_v3.sh` accepts only `status == completed`, pins the scored config, downloads turn-detector weights,
  runs a real TTS->STT speech preflight and asserts judge coverage (`scripts/judge_coverage.py`).
- Integrity audit fixed (it parsed 0 utterances before) and extended with `--strict-comments`: PASS. Offline text replay
  91/100, held-out 29/30.
- Extension = **Triage Line**, routable end to end from `/rtc.html?flow=triage` (closed flow choice ->
  `TRIAGELINE_TRIAGE_AGENT_NAME`). README / submission.yaml / deck agree.

## Still required from the team (needs your keys / hardware — cannot be done in the sandbox)
1. `PYTHON=python3.12 ./run_fdb_v3.sh --limit 5 --require-judge` with LiveKit + speech keys + OPENAI_API_KEY (judge), then the full run. Commit `results/<stamp>/` summary and update `submission.yaml: reported_scores.live_livekit_judged`.
2. Check `results/results.md` per-turn timing; if response median is > 1.5 s, switch speech to Deepgram (streaming) or lower `TRIAGELINE_SETTLE_S`.
3. Live-room checks: talk over TTS, correct mid-sentence, disconnect/rejoin; confirm zero duplicate state changes in `agent_tool_calls.log`.
4. The deck is built (`docs/deck/BILIO_Theme05.pptx` + PDF, 8 slides); rebuild it with the live numbers via `scripts/build_deck.py`. The PWA interruption clip is recorded (`docs/video/pwa_interrupt.mp4`). Screen-record the live FDB-v3 run and a phone call, cut to 3–5 min following `docs/VIDEO_SHOTLIST.md`, and submit via the form.

## Known limits (state them, don't hide them)
Tools are simulated; access code is pilot access, not user identity; sessions are process-local (one replica); web client, not a native SDK app.
