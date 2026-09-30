# BILIO: an interruptible real-time agent (Samsung PRISM · Theme 05) 

BILIO is a dual-process voice agent built on the official Theme 5 harness. A fast path answers within milliseconds. A slow path does the real work (ASR, vision, async tools). A coordination layer handles barge-ins without acting on stale results.

| Public scenario | Baseline | BILIO |
|---|---:|---:|
| pub_01 simple search | 100 | **100** |
| pub_02 interruption (Boston → NYC) | ~90 | **100** |
| pub_03 chained search + booking | low | **100** |
| pub_04 no-tool small talk | ✓ | **100** |
| pub_08 injected timeout, retry | low | **100** |
| pub_09 unseen tool (weather_lookup) | 0 | **100** |
| pub_05 / 06 / 07 audio & visual | 0 | 100 in local tests with synthetic media* |

\*The kit's `audio/` and `frames/` media are **not in this repo**. Copy them in from the participant kit. Without them the agent falls back safely: it acknowledges, then clarifies, and never guesses. I checked pub_06 and pub_07 end-to-end with synthesized TTS clips and a rendered port frame; both scored 100. I checked pub_05's clarify-then-confirm flow by injecting a disagreeing ASR result, not with real audio.

Generated practice scenarios (`harness.scenario_gen`, all templates) and harder hand-written cases in `scenarios_extra/` (retraction, intent switch, unseen read-only tool, unseen state-modifying tool with an enum) all score 100.

## Architecture

```
 events ─▶ FAST PATH  (<5 ms, inline)   turn buffer · self-repair resolver ·
           interruption classifier (revise / retract / switch) · content-aware ack · snapshot
        ─▶ SLOW PATH  (asyncio tasks)   ASR ensemble · OCR+CLIP frame analysis ·
           schema-driven tool args · 1× read-only retry · chained plans
        ─▶ COORDINATION                  in-flight ledger → targeted cancel_tool ·
           epoch counter → stale results dropped · idempotence ledger for state-modifying calls
```

Key ideas:

1. **Interruption taxonomy.** Each barge-in is classified as a slot revision, a retraction or an intent switch, and each type has its own cancel-and-replan policy.
2. **Epoch-guarded grounding.** Each interruption increments an epoch, and anything started in an older epoch is ignored. This also covers the case where a cancel races with a result.
3. **Uncertainty-aware ASR.** `base.en` and `tiny.en` decode independently. If they disagree, or the word probability on a slot value is below 0.8, the agent asks "did you say X or Y?" using what was actually heard. Clips are transcribed as they stream in; the first acknowledgement is spoken before ASR finishes.
4. **Zero-shot tools from the manifest.** Tools are ranked by lexical plus schema overlap, with a bonus when every required argument can be filled. Arguments are built by what each schema field means (city, date, name, enum, number, nested object). Answers are grounded in whatever fields the result returns, including lists of records.
5. **Visual grounding.** Printed-label OCR (rotated and inverted passes) is fused with CLIP zero-shot on a centre crop. The 512-d CLIP embedding goes to `lookup_manual` for hybrid search. If vision is unsure, the agent asks instead of guessing.
6. **Safety by construction.** At most 3 fillers, no repeated filler text, and state-modifying calls deduplicated by argument hash. Replies promise actions and only claim them once done. Every spoken action carries a `state_snapshot`.

The agent uses no scenario ids, timestamps or expected strings, and never reads `ground_truth` or `_` annotations.

## Run

```bash
pip install -r requirements.txt                   # optional models; agent runs without them
python run_local.py --all --agent agent.agent:ParticipantAgent          # official scale 1.0
python run_local.py --scenario scenarios_extra/x_intent_switch.json --agent agent.agent:ParticipantAgent
python ui/server.py                                # console at http://localhost:8080
```

Environment knobs: `ASR_MODEL` (defaults to `small.en` on GPU, `base.en` on CPU), `ASR_BEAM`, `ASR_ENSEMBLE=0`, `CLIP_REPO`.

## Console (`ui/`)

A zero-dependency web UI on top of the real harness and scorer:

- **Run**: any scenario with either agent. Shows a lane timeline (user, fast path, slow-path tool bars with cancellations, answer), the transcript with slot snapshots, and every scorer checkpoint.
- **Compose**: write your own utterance and barge-in, optionally with an unseen tool, and watch the agent handle it.
- **Suite**: baseline vs BILIO across all scenarios.

## Layout

```
agent/agent.py        ParticipantAgent (entry point) — orchestration
agent/nlu.py          fast-path NLU: slots, repair, routing, schema-driven args
agent/perception.py   slow-path ASR + vision (module-level cached, loaded in setup())
agent/baseline_agent.py  reference agent from the kit
harness/, docs/, scenarios/   official kit (unchanged)
scenarios_extra/      harder hand-written probes
ui/                   console (server.py + static/)
legacy/               earlier Phase-7 prototype (kept for history)
submission.yaml
```
