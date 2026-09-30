# Demo video shot list (3–5 min) — record only what actually ran

Label every simulated action as SIMULATED on screen. Never show a failed or old-kit run as FDB-v3.

1. **0:00–0:30 Problem + architecture.** One slide: VAD → STT → ParticipantAgent (fast path acks, slow path tools, ledger) → TTS over LiveKit.
2. **0:30–1:45 Real FDB-v3 interruption.** Terminal: `./run_fdb_v3.sh --limit 5 --require-judge` completing successfully; show one example's `result_bilio.json` with a self-correction and the corrected tool call; show the judged report line.
3. **1:45–2:10 PWA interruption (ready-made).** `docs/video/pwa_interrupt.mp4` (regenerate with `python3 scripts/record_ui_demo.py`): request → barge-in correction → stale flight search cancelled → corrected booking → same-breath retraction performs nothing.
4. **2:10–3:15 Extension, end to end.** Phone on `/rtc.html` (HTTPS): connect, ask for a SIMULATED booking, interrupt the assistant mid-sentence with a correction, show audio stopping and the corrected result; lock and unlock the phone (mic pauses/resumes, call survives).
5. **3:15–4:00 Safety.** Assistant mode reads back an LLM-proposed state change and waits for "yes"; a same-breath "actually, don't do that" performs nothing.
6. **4:00–4:30 Honest numbers.** Live judged score (from results/results.md), held-out 26/30 rules-only, integrity audit PASS; offline text replay labelled "diagnostic".

Slides: `docs/deck/BILIO_Theme05.pptx` (8 slides). After the live run, rebuild with `python3 scripts/build_deck.py --live-score "NN/100" --live-latency "N.N s"`.
