# LiveKit Agent — Technical Audit (independent of benchmark runner)

Scope: is `livekit_agent/` a genuine, wired-together implementation, or scaffold?
This is a code-level audit. LiveKit/OpenAI credentials and network egress were
NOT available in this session; everything below is labeled accordingly.

## 0. Top-line finding

**There are three separate, unconnected agent implementations in this repo,
not one integrated pipeline.** None of them is exactly the
"LiveKit → cascaded_agent → adapter → existing coordination logic → tools"
chain the brief describes.

| Path | Entrypoint | Talks to | Has real epoch/interrupt/dedup logic? |
|---|---|---|---|
| A | `cascaded_agent.py` | `mock_apis.py` directly (LLM tool-calling) | **No** — no adapter/agent.agent import at all |
| B | `triage_livekit_agent.py` | `triage_brain.py` → `legacy/core/*` | Yes, but via a *different* engine (DialogueEngine/CommitStateMachine), not `agent.agent.ParticipantAgent` |
| C | `adapter.py` + `fdb_tools.py` | `agent.agent.ParticipantAgent` (the epoch/dedup engine referenced in the brief) | Yes — but only exercised by `fdb_scenario_run.py`, a standalone script. `attach_livekit_session()` (the function that would wire it into a live LiveKit session) is defined but **never called anywhere in the repo.** |

So the specific chain the audit asked about — LiveKit session → cascaded_agent
→ adapter → ParticipantAgent → tools — **does not exist as running code**. Path
A (the thing that actually launches under `python cascaded_agent.py`) has no
epoch/interruption/dedup logic. Path C (the thing with real epoch/interruption/
dedup logic) has no LiveKit session driving it.

---

## 1. LiveKit agent reality (`cascaded_agent.py`)

| Item | Status | Evidence |
|---|---|---|
| Real `livekit.agents` imports | UNVERIFIED | `from livekit import agents, rtc`, `from livekit.agents import Agent, AgentSession, AgentServer, llm` — correct-looking API, but `livekit` is not installed in this sandbox (`ModuleNotFoundError: No module named 'livekit'`), so this file **cannot even be import-checked here**. SETUP.md claims it was verified to import against `livekit-agents 1.8.3` in the authoring sandbox — that claim is UNVERIFIED by me, not confirmed false. |
| Agent/AgentSession/entrypoint structure | REAL (as written) | `class CascadedVoiceAgent(Agent)`, `server = AgentServer()`, `@server.rtc_session()` entrypoint, `AgentSession(vad=..., stt=..., llm=..., tts=..., tools=...)`, `session.start(room=ctx.room, agent=...)` — this is a coherent, idiomatic LiveKit Agents 1.x shape, not a stub. |
| STT integration | REAL (as written) / UNVERIFIED (runtime) | `openai.STT(model="whisper-1", language="en")`. |
| LLM integration | REAL (as written) / UNVERIFIED (runtime) | `openai.LLM(model="gpt-4o")`, tools passed via `llm.find_function_tools(fnc_ctx)`. |
| TTS integration | REAL (as written) / UNVERIFIED (runtime) | `openai.TTS(model="tts-1", voice="nova")`. |
| Event/callback handling | REAL (as written) | `session.on("user_input_transcribed")`, `session.on("agent_state_changed")` — both used for latency tracking, not decoration. |
| Env-var usage | REAL | `.env.local` loaded via `dotenv`; script checks `LIVEKIT_URL/API_KEY/API_SECRET/OPENAI_API_KEY` (checked in `run_fdb_v3.sh`, not in this file itself). |
| **Adapter / epoch / interruption logic** | **MISSING from this file** | No import of `adapter.py`, `agent.agent`, or `fdb_tools.py`. Tool functions call `registry.call(...)` (mock_apis) directly. Barge-in/interruption handling is whatever `AgentSession`'s own default `allow_interruptions` does — nothing custom. |

**Verdict for Section 1:** the LiveKit *scaffolding* is REAL (a competent, idiomatic AgentSession setup), but it is functionally a vanilla LLM-tool-calling voice agent with no connection to the interruption/epoch/dedup system the competition rubric is asking about. Whether it actually runs is UNVERIFIED (no `livekit` package, no credentials, no network here).

---

## 2. `adapter.py`

`adapter.py` imports cleanly with no external deps beyond `agent.agent` (no `livekit` needed), so this part *is* testable locally, and I ran the checked-in tests myself (see §7).

**Call flow it implements** (traced from source, `livekit_agent/adapter.py:117-227`):

```
LiveKit "user_input_transcribed" (is_final=True)
  → TriageAdapter.on_user_final(text)
      if agent busy (inflight call or turn unanswered):
        → on_barge_in(text)
          → in_q.put({"interruption", text})
            → ParticipantAgent.on_interruption()   [agent/agent.py:595]
              → .revise() or .retract() or switch classification
                → self.version += 1                 (epoch bump, agent/agent.py:655)
                → cancel_where(pred)                 [agent/agent.py:232]
                  → matching in-flight calls marked cancelled;
                    self.ops[key]["status"] = "cancelled"
                  → out_q.put({"cancel_tool", call_id})
                    → TriageAdapter._pump_outputs() → self._tool_canceller(call_id)
      else:
        → in_q.put({"user_speech_chunk", end_of_turn=True})   (ordinary new turn)
```

Tool issuance side:
```
ParticipantAgent.call(api, args)                     [agent/agent.py:197]
  → op_key = api + canonical-json(args)               (dedup key)
  → if op already pending/done for this key: short-circuit, don't re-issue
  → else: self.inflight[cid] = {..., "version": self.version}
    → out_q.put({"tool_call", call_id, api, args})
      → TriageAdapter._pump_outputs() → self._tool_executor(call_id, api, args)
```

Result delivery:
```
adapter.on_tool_completed(call_id, result, status)
  → in_q.put({"tool_result", call_id, result, status})
    → ParticipantAgent drops it silently if call_id was already popped by
      cancel_where() — i.e. a cancelled call's late result is never grounded on.
```

| Item | Status | Evidence |
|---|---|---|
| Interruption detection/handling | **REAL** | `on_user_final`'s `busy` check + `on_interruption` in `agent/agent.py`; exercised by `adapter_tests/test_1` and `test_2` (§7), which show `epoch before/after` incrementing and `cancelled` lists populated. |
| Epoch/version tracking | **REAL** | `self.agent.version`, bumped in `.invalidate()`/`.revise()` (`agent/agent.py:246,655`), read at call time (`agent/agent.py:224`). |
| Stale-work cancellation | **REAL** | `cancel_where()` + the fact that `on_tool_completed` drops results for popped `call_id`s — confirmed by test 2's `cancelled: ['c1']` while `c2` proceeds. |
| Updated-intent handling (revise) | **REAL** | `.revise()` in `agent/agent.py`, exercised by test 1 (Boston→Chicago). |
| Tool-argument replacement | **REAL** | New tool call `c2` in test 1/2 carries the corrected args, not the stale ones. |
| Deduplication | **REAL** | `op_key`/`self.ops` ledger; exercised by `adapter_tests/test_3_duplicate_state_change.py` — a second `book_flight` request with identical args does **not** issue a second call. |
| Async tool execution | **REAL** | `_tool_executor` is awaited from `_pump_outputs`, called concurrently via `asyncio.Queue`, not sequential blocking calls. |
| **Integration with the actual LiveKit runtime** | **UNWIRED** | `attach_livekit_session()` is the only function in this file that would connect it to a real `AgentSession`. It is defined (`adapter.py:191-226`) but **grep of the whole repo finds zero callers** of `attach_livekit_session` or `TriageAdapter` outside `adapter.py` itself, its own tests, and `fdb_scenario_run.py` (a standalone scenario script, not a LiveKit entrypoint). `cascaded_agent.py` and `triage_livekit_agent.py` — the two files that actually run under LiveKit's CLI — import neither `adapter.py` nor `TriageAdapter`. |

**Verdict for Section 2:** the interruption/epoch/dedup *logic* is real, tested, and reused (not reimplemented) — but it is currently **UNWIRED** to LiveKit. It only runs today inside a hand-written scenario harness that fakes the LiveKit callback shapes.

---

## 3. `fdb_tools.py`

- **IMPLEMENTED: 12/12** — the file defines exactly 12 tools across 4 domains (Travel & Identity / Finance & Billing / Housing & Location / E-Commerce Support), and asserts `len(FDB_TOOLS) == 12` at import time.
- **VERIFIED: 12/12 schemas are internally consistent and match `mock_apis.py`'s registry** (same names/kwargs) and `cascaded_agent.py`'s `AssistantFnc` methods — I diffed the tool names in both files and they match.
- **VERIFIED against the *official* FDB-v3 tool definitions: 0/12.** The file's own docstring says this directly: the official FDB-v3 difficulty tags and scenario definitions are "NOT available in this sandbox (no network, no FDB Google-Drive data bundle)... NOT invented here." So the 12 tools are a plausible, self-consistent reconstruction, not a confirmed match to the real benchmark's tool set. That's an honest gap, not a fabrication — but it is a real gap.
- Chained calls: exercised in `fdb_scenario_run.py`'s `chained:*` scenarios — **1 of 2 passed** (`step2_book_flight_followed` PASS, `step1_search_flights_issued` FAIL — see §7).
- Cancellation: REAL, see §2/§7 (test 1, test 2).
- Duplicate state-changing calls prevented: REAL, see §2/§7 (test 3, and the scenario runner's `dedup:duplicate_add_to_cart_blocked_while_pending` PASS).
- Async: REAL — tools are `async def` in `cascaded_agent.py` and issued through an `asyncio.Queue` in `adapter.py`.

---

## 4. Interruption trace: "hotel in Chennai" → "actually, make that Bangalore"

Running this scenario requires either (a) the real agent under LiveKit (blocked — no credentials/network) or (b) the adapter path. I did **not** find a checked-in scenario using these exact cities, so I traced the mechanism using the equivalent checked-in scenario (`test_1_interrupt_while_reasoning.py`: "flights to Boston" → "actually Chicago") which is structurally identical:

| Step | Happens? | Evidence |
|---|---|---|
| 1. epoch increments | **YES** | `epoch before: 0 epoch after: 1` (test output) |
| 2. old task cancelled | **YES** | `cancelled: ['c1']` |
| 3. cancellation reaches the async task | **YES, at the call-tracking level** | `c1` is marked cancelled in `self.ops`/`cancel_where`; `out_q` emits a `cancel_tool` action for it |
| 4. stale result prevented from emitting | **YES** | `on_tool_completed` silently drops results for a `call_id` no longer in `inflight`/already cancelled (code-level; not exercised with an actual late-arriving result in the test, so this is inferred from the drop-path code plus the dedup test's behavior, not directly logged) |
| 5. new arguments generated | **YES** | `c2` carries `destination: 'Chicago'` |
| 6. new tool call uses new city | **YES** | Same evidence |
| 7. old action cannot execute afterward | **YES, in this harness** | `c1` never appears again in issued/spoken output after cancellation |
| 8. duplicate state-changing action prevented | **Tested separately, not in this scenario** | Confirmed by `test_3_duplicate_state_change.py`, not by test 1/2 |

**Caveat:** all of this is verified through `adapter.py`'s queue-based harness, i.e. Path C. It has **not** been verified through the actual `cascaded_agent.py` LiveKit entrypoint (Path A), because Path A doesn't call into this logic at all (§0/§2).

---

## 5. Fast path / slow path

- `adapter.py`'s docstring explicitly states user-speech **partial** transcripts (`on_user_partial`) are "intentionally a no-op today" — there is no fast conversational backchannel wired to partial STT.
- What does exist: `_pump_outputs()` speaks `filler_speech` (e.g. "On it — finding flights to Boston first.") immediately when a tool call is issued, while the tool call itself proceeds asynchronously via `asyncio.Queue` + `asyncio.create_task`-style dispatch. That is a genuine fast/slow split — filler speech is not blocked on tool completion — but:
  - It is issued from the **fast path already inside `agent.agent.ParticipantAgent`**, not something `adapter.py` or `cascaded_agent.py` add.
  - It runs concurrently in the sense that `_pump_outputs` and the agent's own `run()` loop are separate `asyncio.Task`s reading/writing shared queues — this is real `asyncio` concurrency, not sequential simulation, **within Path C**.
  - **Path A (`cascaded_agent.py`, the one that actually launches)** has none of this — it relies entirely on whatever backchannel/filler behavior `AgentSession`'s defaults provide, which is standard LiveKit LLM-agent behavior, not this project's fast/slow-path design.

---

## 6. Failure / race-condition scan

Checked by reading `agent/agent.py`, `adapter.py`, and by running the existing scenario/adapter tests (§7):

- **Cancelled tasks continuing to execute:** not observed in the tests run; `cancel_where()` marks ops cancelled synchronously before the cancel signal is emitted, so no window was seen. Not stress-tested for genuine `asyncio.Task.cancel()` races (the harness here simulates cancellation at the bookkeeping level; whether a real in-flight `asyncio.Task` wrapping an HTTP call actually gets `.cancel()`'d promptly is UNVERIFIED — `mock_apis.py` calls are fast/local, so a real race would need a slow tool call to expose).
- **Stale results emitted:** not observed; drop-on-stale-call_id logic exists (§2).
- **Duplicate tool calls:** not observed; dedup ledger works in all 3 dedup-relevant tests/scenarios.
- **Cancellation swallowed by exception handlers:** `TriageAdapter._pump_outputs()` catches only `asyncio.CancelledError` and returns — doesn't swallow other exceptions, but also doesn't log them (a bare `except` isn't present, so this is fine, just silent on the `CancelledError` path specifically, which is the intended behavior for shutdown).
- **Tasks not awaited:** `TriageAdapter.stop()` does `t.cancel()` then `await t` for both `_pump_task` and `_run_task` — correctly awaited.
- **Tasks leaked after teardown:** `triage_livekit_agent.py` registers `_teardown` against both `room.on("disconnected", ...)` and `ctx.add_shutdown_callback(...)`, and `test_4_triage_line_interruption.py` explicitly verifies `force_resolve_pending on teardown` resolves a pending action rather than leaving it dangling. This is real and tested (Path B, not Path A or C, though).
- **Rapid successive interruptions:** not covered by any checked-in test I found (`test_1`/`test_2` each apply exactly one interruption). This is a genuine untested gap.

---

## 7. Local test execution — actual results

Environment constraints hit immediately: **no network egress, `pytest` is not installed and cannot be installed** (`pip install pytest` failed — `No matching distribution found`). `tests/test_regressions.py` and `legacy/tests/*` are written for pytest specifically and are not `unittest.TestCase`-based, so `python -m unittest discover` finds 0 tests in them. **These suites are UNVERIFIED in this session, through no fault of the code** — I could not execute them here.

What I *could* and did run directly (no external deps beyond the stdlib + this repo):

```
$ python3 livekit_agent/adapter_tests/test_1_interrupt_while_reasoning.py
epoch before: 0 epoch after: 1
cancelled: ['c1']
PASS: test_1_interrupt_while_reasoning

$ python3 livekit_agent/adapter_tests/test_2_interrupt_while_tool_running.py
epoch before: 0 epoch after switch: 1
cancelled: ['c1']
PASS: test_2_interrupt_while_tool_running

$ python3 livekit_agent/adapter_tests/test_3_duplicate_state_change.py
PASS: test_3_duplicate_state_change

$ python3 livekit_agent/adapter_tests/test_4_triage_line_interruption.py
Step 6 OK — force_resolve_pending on teardown resolved: [...]
PASS: test_4_triage_line_interruption

$ python3 livekit_agent/fdb_scenario_run.py
PASS single_call:search_flights
PASS single_call:get_card_benefits
FAIL single_call:search_apartments   — issued=[] (tool never called)
PASS single_call:track_order
PASS single_call:search_products
FAIL chained:step1_search_flights_issued — issued=[('c1','book_flight',...)] (step 1 tool missing from issued list)
PASS chained:step2_book_flight_followed
PASS interruption:stale_cancel_updated_args
PASS dedup:duplicate_add_to_cart_blocked_while_pending
7/9 scenarios PASS.
```

**This matches exactly** what's already checked in at `livekit_agent/logs/test_summary.md` — i.e. the committed test-summary claim (7/9 pass, 2 named failures) is **CODE VERIFIED LOCALLY**, not a stale or inflated claim.

- **CODE VERIFIED LOCALLY:** all 4 `adapter_tests/*`, the `fdb_scenario_run.py` 9-scenario suite (7/9 pass, 2 fail as listed above), `agent/agent.py`'s epoch/dedup mechanism (exercised indirectly through the above).
- **REQUIRES LIVEKIT/NETWORK/CREDENTIALS (not run):** `cascaded_agent.py`, `triage_livekit_agent.py` (both fail to even `import` here — `ModuleNotFoundError: No module named 'livekit'`), the official FDB-v3 benchmark, `tests/test_regressions.py` / `legacy/tests/*` (blocked on missing `pytest`, not on credentials — worth fixing regardless of network state).
- **NOT IMPLEMENTED:** `attach_livekit_session()` is never called; rapid-multi-interruption handling has no test; partial-transcript fast path is an explicit no-op.

---

## 8. Architecture verdict

> "Is `livekit_agent/` actually connected to the existing BILIO intelligence, or is it currently a standalone wrapper?"

**Neither cleanly.** It's split:
- `triage_livekit_agent.py` **is** connected to the existing BILIO intelligence (`legacy/core`'s DialogueEngine + CommitStateMachine) via `triage_brain.py`, and that connection is real and tested (`test_4`).
- `cascaded_agent.py` (the file the audit brief and `run_fdb_v3.sh` treat as *the* FDB-v3 agent) is **not** connected to either the BILIO engine or to `agent.agent.ParticipantAgent`'s epoch/dedup system. It's a standalone LLM-tool-calling wrapper around `mock_apis.py`.
- The one piece that bridges LiveKit-shaped events to `agent.agent.ParticipantAgent` (`adapter.py`) is real and tested but **not wired into any process that LiveKit actually launches.**

So the specific chain LiveKit → cascaded_agent → adapter → existing coordination logic → tools is **NOT a real execution path today.** Two of its four links (`cascaded_agent` → `adapter`, and `adapter`'s LiveKit-facing half) don't exist as live connections; the other two links (`adapter` → `ParticipantAgent`, and `ParticipantAgent` → tools) are real but only reachable from a standalone scenario script.

---

## 9. Subsystem scores (internal engineering only — NOT the competition score)

| Subsystem | Score /10 | Why |
|---|---|---|
| LiveKit runtime integration | **4/10** | Two idiomatic, well-formed LiveKit `AgentSession` entrypoints exist (Path A, Path B) but neither could be import-verified here (no `livekit` package), and SETUP.md itself says only "imports cleanly" was checked, not a live room join. |
| Interruption implementation | **7/10** | Genuinely real, epoch-based, tested (§2, §4) — but only inside Path C, which is disconnected from the actual LiveKit runtime. Rapid multi-interruption untested. |
| Async/cancellation | **6/10** | Real `asyncio` queue-based concurrency and correct task-cancellation/await patterns where exercised (§6) — but never stress-tested against a slow/real tool call, only fast local mocks. |
| Tool integration | **6/10** | 12/12 tools defined and internally consistent across 3 files; 7/9 hand-written scenarios pass; 2 concrete failures exist and are logged, not hidden. 0/12 verified against the *official* FDB-v3 tool set (data unavailable, honestly disclosed). |
| Code integration/architecture | **3/10** | This is the weakest link: three parallel agent implementations, only one pair of which (`adapter.py`↔`ParticipantAgent`) is actually tested together, and it's the pair not reachable from a real LiveKit session. The competition-relevant chain doesn't exist as running code yet — it exists as two well-built halves that were never joined. |

---

## 10. P0/P1/P2 fixes (audit only — nothing below has been implemented)

### P0 — fundamentally non-functional/missing
1. **`cascaded_agent.py` does not use `adapter.py`/`ParticipantAgent` at all.**
   File: `livekit_agent/cascaded_agent.py`.
   Problem: the only file that actually launches as a LiveKit agent has zero epoch/interruption/dedup logic — it's a bare LLM tool-caller.
   Why it matters: this is exactly what FDB-v3 scores (interruption handling, stale-intent cancellation, duplicate-action prevention) — currently unimplemented in the one process a judge would run.
   Fix direction: replace `AssistantFnc`'s direct `registry.call(...)` tool bodies with calls into a `TriageAdapter` instance; wire `session.on("user_input_transcribed")` to `adapter.on_user_final`/`on_barge_in` via `attach_livekit_session()` instead of hand-rolled logging-only handlers.

2. **`attach_livekit_session()` is dead code.**
   File: `livekit_agent/adapter.py`.
   Problem: the one function that would connect the tested interruption logic to a live `AgentSession` has zero callers anywhere in the repo.
   Why it matters: without this call, item 1 above cannot be fixed by wiring alone — this function needs to actually be invoked from an entrypoint.
   Fix direction: call `attach_livekit_session(session, adapter, room_name=ctx.room.name)` inside `cascaded_agent.py`'s (or a merged) `entrypoint()`.

3. **Benchmark runner cannot reach Stage 5** (already covered in the previous audit) — not re-scored here, but blocks proving any of the above works end-to-end against the real FDB-v3.

### P1 — functional but incomplete
4. **2 of 9 local scenarios fail.** Files: `livekit_agent/fdb_scenario_run.py` + whichever of `agent/nlu.py`/`agent/agent.py` handles `search_apartments` arg-filling and the first step of the chained-call scenario. `search_apartments` never gets issued at all (`issued=[]`), and the chained scenario's step-1 tool call is missing from the issued list. Fix direction: debug `agent/nlu.py`'s slot-filling for `search_apartments`'s 3 required args, and check why the chained scenario's first call isn't being tracked/issued before step 2 fires.
5. **Rapid successive interruptions untested.** File: new test alongside `livekit_agent/adapter_tests/`. Fix direction: add a test sending 3+ interruptions in quick succession before any tool completes, and confirm only the final epoch's call survives.
6. **`tests/test_regressions.py` / `legacy/tests/*` can't run in a no-network judging sandbox** because they require `pytest`, which isn't vendored. Fix direction: either vendor a pytest wheel in the repo, or add a stdlib-`unittest`-compatible runner/shim so judges without network access can still execute the regression suite (this is a real, avoidable reproducibility risk independent of FDB-v3 data access).

### P2 — robustness/polish
7. Partial-transcript fast path (`on_user_partial`) is a documented no-op — fine to leave as-is for the competition, but worth explicitly noting in submission docs as a known simplification rather than leaving it only in a code comment.
8. Cancellation of a genuinely slow/real (non-mock) tool call has never been exercised — worth one test with an artificial `asyncio.sleep()` tool to confirm `.cancel()` actually interrupts mid-flight rather than just being marked cancelled in bookkeeping.

---

## Component summary table

| Component | Status | Evidence |
|---|---|---|
| LiveKit runtime | PARTIAL / UNVERIFIED | Well-formed `AgentSession` code in 2 files; cannot import here (`no module named 'livekit'`); SETUP.md itself only claims "imports cleanly," not a live join. |
| STT | PARTIAL / UNVERIFIED | `openai.STT(model="whisper-1")` wired in code; never run live here. |
| LLM | PARTIAL / UNVERIFIED | `openai.LLM(model="gpt-4o")` wired in code; never run live here. |
| TTS | PARTIAL / UNVERIFIED | `openai.TTS(model="tts-1")` wired in code; never run live here. |
| Adapter | REAL (in isolation) / UNWIRED (in practice) | `adapter.py` logic is real and tested; never called from a live LiveKit entrypoint. |
| Interruption | REAL (in Path C only) | `adapter_tests/test_1`, `test_2` pass locally, epoch increments confirmed. |
| Cancellation | REAL (in Path C only) | Same tests; `cancel_where` + drop-stale-result logic confirmed. |
| Epoch handling | REAL | `agent/agent.py` `self.version`, exercised in tests. |
| Deduplication | REAL | `adapter_tests/test_3` passes; op-ledger confirmed. |
| FDB tools | PARTIAL | 12/12 defined and internally consistent; 0/12 confirmed against the official FDB-v3 set; 7/9 local scenarios pass. |
| Chained calls | PARTIAL | 1 of 2 chained scenarios passes; the other fails (`step1_search_flights_issued`). |
| Fast/slow path | PARTIAL | Real concurrency exists in Path C (filler speech vs. tool execution); Path A (the actual launch entrypoint) has none of it. |
| BILIO integration | PARTIAL, SPLIT | Real in Path B (`triage_livekit_agent.py` ↔ `legacy/core`, tested via `test_4`); absent in Path A; unreachable-from-LiveKit in Path C. |

## OVERALL VERDICT: **PARTIALLY REAL**

The interruption/epoch/dedup engine and the BILIO deliberation engine are both genuinely implemented and locally tested — this is not scaffold dressed up as substance. But the specific integration the competition (and this audit) cares about — a single LiveKit agent that runs the FDB-v3 tool set *through* the tested interruption/dedup logic — does not exist as one running path. It exists as two tested-but-separate halves plus a third, unrelated LLM-tool-calling agent that is the one thing actually reachable via `python cascaded_agent.py`. Closing that gap (P0 items 1–2 above) is the single highest-leverage fix available before worrying about FDB-v3 data/credentials at all.
