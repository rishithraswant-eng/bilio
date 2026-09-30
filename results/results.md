# FDB-v3 results — BILIO

Generated: 2026-09-30T17:07:07+00:00

Run directory: `results\20260930T170510Z`  
Mode: **offline_text_replay**  (official data + official evaluators; no LiveKit transport, no audio latency)
Provider name: `bilio_text` · LLM judge: **off (exact match = lower bound)** · examples: 100  
FDB-v3 commit: `3e799c45a045256f47d5f1c9cda90157e2d2ec9e` · BILIO commit: `217fc2eeb4ab03573c09cc8fae432b7f60eb7e23` · 3.12.10  
Agent: custom LiveKit agent (Silero VAD → STT → ParticipantAgent: rules only (LLM planner off) → TTS) · STT=deepgram:nova-3 TTS=deepgram:aura-2-andromeda-en

> **Diagnostic only.** Offline text replay feeds the official transcripts to the agent without LiveKit, STT or TTS. It is **not** the scored live FDB-v3 run and must not be reported as one.

## Headline (official evaluators)

| strict pass rate | tool-selection acc | argument acc | response quality | turn-take rate |
|---|---|---|---|---|
| **61.0%** (61/100) | 89.8% | 68.9% | not produced | 100.0% |

### Dev vs hash split of the public set (`livekit_agent/fdb_split.py`)

Both halves come from the same public benchmark that rules were developed against, so this is **not** an independent held-out set. See `scenarios_heldout/` for the independent paraphrase set.

| split | passed | rate |
|---|---|---|
| dev | 29/48 | 60.4% |
| heldout | 32/52 | 61.5% |

**By domain:** ecommerce_support 72.0% · finance_billing 68.0% · housing_location 44.0% · travel_identity 60.0%

**By disfluency:** FILLER 65.4% · PAUSE 52.6% · HESITATION 60.0% · FALSE_START 72.7% · SELF_CORRECTION 57.1%

**By difficulty:** easy 65.6% · medium 69.4% · hard 46.9%

**By number of tools:** 1 66.1% · 2 47.8% · 3 64.3% · 4 0.0%

**Failures:** wrong_tools=25, wrong_arguments=14

## Latency

Not measured in offline mode (no audio). Run the full `./run_fdb_v3.sh` for latency.

Files: .inference_started, bilio_text_evaluation_report.json, bilio_text_pass_rate_report.json, pip_freeze.txt, run.log, run_config.json
