# BILIO — Full Audit vs Theme 05 Guidelines (2026-09-25)

> Scope: this is an analysis only. **No project code was changed.** Every number below comes from a
> command run in this session (see §9, "How to reproduce this audit"). If something could not be
> verified, the report says so.
>
> Guides checked: `Theme05_Participant_Guide_UPDATED_FBD.docx` (**the scored rules**: LiveKit +
> official FDB-v3) and `Theme 5_Guide.pdf` (the older kit: 9-scenario harness, still useful for
> interruption behaviour but **not scored**).

---

## 0. Executive summary

| Area (weight) | Status | Main reason |
|---|---|---|
| **Benchmark, 60%** (organizers re-run your script) | 🔴 **Would currently score ~0** | `run_fdb_v3.sh` is hard-coded to fail at Stage 3 or Stage 5. It never runs inference or evaluation. Even when it is fixed, the agent's official strict pass rate is **16%** (text mode) and **15%** (audio mode). |
| **Extension, 20%** | 🟠 Partial | Triage Line only works offline with mocks. It has never run in a live LiveKit room and is not in any video. Its confirmation safety has real bugs ("Do not confirm" → it dispatches). |
| **Docs / architecture / video, 20%** | 🟠 Partial | There is no demo video and no slide deck (only outlines). The README declares the wrong model provider (gpt-4o, but the agent has no LLM). Several README claims are no longer true. |

**Five things to fix first, in this order:**
1. Make `run_fdb_v3.sh` actually run inference + the three official evaluators end to end (§2, B-01).
2. Stop the LiveKit adapter from blocking speech and cancellation while a tool runs (§3, B-05).
3. Fix the latency telemetry so the official runner can read it (§3, B-06/B-07).
4. Fix argument extraction and self-correction, the main source of lost score (§4).
5. Record the video and build the ≤8-slide deck. Run the extension live (§6).

---

## 1. What was tested and the results

| # | Test | Result |
|---|---|---|
| T1 | `pytest tests` (root) | ✅ 81 passed |
| T2 | `pytest legacy/tests` | ✅ 209 passed (README still says 191/191) |
| T3 | `livekit_agent/adapter_tests/test_1..6` | ✅ 6/6 pass |
| T4 | `run_local.py --all` (old kit, not scored) | ✅ 89.1/100 (pub_05 53.8, pub_06 56.9, pub_07 90.8, rest 100). Matches the README. |
| T5 | Import `cascaded_agent.py` and `triage_livekit_agent.py` with **livekit-agents 1.8.3** | ✅ both import; all event names they use exist in 1.8.3 |
| T6 | `pip install livekit-agents`, `git clone` FDB-v3, Google-Drive data via `gdown` | ✅ **all work**. The README/readiness claim of "no PyPI egress" is out of date. |
| T7 | Official FDB-v3 data (100 examples) → `fdb_v3_offline_replay.py --text` → **official** `evaluate_pass_rate.py` / `evaluate_tool_calls.py` | 🔴 tool-name match 57/100 · **strict pass 16.0%** · tool-selection acc 65.6% · **argument acc 21.5%** |
| T8 | Same, **audio** (faster-whisper base.en), 20-example stratified subset | 🔴 tool-name match 12/20 · **pass 15%** (the same 20 in text mode: 10%) |
| T9 | Same audio, 0.55 s turn gap (like LiveKit `min_silence_duration`) | 🔴 11/20 · pass 15% · 5/20 utterances split into several turns |
| T10 | Concurrency probe: slow tool (1 s) + interruption after 100 ms | 🔴 the cancel and the "switching" line come out only **after the stale tool finishes** (≈1000 ms late) |
| T11 | Triage confirmation-safety probes | 🔴 4 unsafe dispatches, 1 duplicate dispatch, 1 unhandled exception (§5) |
| T12 | ruff (F, E9, B, ASYNC, BLE) / bandit (-ll) | 41 lint findings; 6 medium bandit findings (§7) |
| T13 | UI server smoke (`/api/health`, `/live.html`, live session, path traversal) | ✅ works; traversal blocked; CORS `*` + 0.0.0.0 bind (low risk) |

**LLM judge:** the official judge needs gpt-4o. The sandbox key only exposes other models and has no
credits, so the evaluator fell back to exact match (a lower bound). Normalising the obvious format
differences (ordinals, `_`/space, hyphens) raises the text-mode estimate to about **28%**. That is
roughly the upper bound for what the judge would forgive. The remaining ~72% are real errors.

Per-scenario results for all 100 examples: `docs/audit/FULL_AUDIT_2026-09-25_appendix_per_scenario.md`.

---

## 2. Benchmark reproduction (60% of the score): blockers

### B-01 🔴 P0 — `run_fdb_v3.sh` can never succeed
- `stage_3_fetch_fdb_data`: after a successful clone it **always** calls `fail` (the last line of the function). The data download is not automated, but `gdown 1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz` works (736 MB, about 10 s).
- `stage_5_run_fdb_eval`: this is just `fail "...Wire the actual command..."`. It never runs `run_tool_benchmark_all_released.py` or the evaluators.
- The guide says: *"if your script does not reproduce … this portion scores zero."* **Right now it scores zero.**

**Fix (write it exactly like this):**
1. Stage 1: create a venv with **Python 3.10–3.12** and pin every dependency (livekit-agents, plugins, livekit, python-dotenv, numpy, openai, pydub, ffmpeg-python, nemo_toolkit[asr], gdown, faster-whisper, …). Commit a `requirements-fdb.lock`. Check that `ffmpeg` is present.
2. Stage 2: require `LIVEKIT_URL/KEY/SECRET` and `OPENAI_API_KEY` (the judge needs it, and so does the STT/TTS in this cascade). Write `v3/.env.local` from them. Never write secrets into `results/`.
3. Stage 3: clone FDB-v3 **pinned to a commit** (`3e799c45a045256f47d5f1c9cda90157e2d2ec9e` was HEAD during this audit). Run `gdown` for the data, unzip to `v3/fdb_v3_data_released`, and drop `__MACOSX/`. Skip this if the data already exists.
4. Stage 4: copy `mock_apis.py`/`latency_injector.py` from the pinned clone. Start `python livekit_agent/cascaded_agent.py start` in the background. Wait until the log says it is registered (not `sleep 2`), with a timeout.
5. Stage 5: `cd v3 && python run_tool_benchmark_all_released.py --provider bilio --root_dir fdb_v3_data_released`, then the three evaluators **with `--use-llm`** (`evaluate_tool_calls.py`, `evaluate_pass_rate.py`, `analyze_tool_latency.py`), all with `--provider bilio`.
6. Stage 6: copy the three JSON reports + every `result_bilio.json` + the agent log + `pip freeze` + the git SHAs into `results/<timestamp>/`, and update `results/results.md` from them.
7. Add `--limit N` for a smoke run and `--skip-install`. Make the script idempotent. Kill the agent in the `trap`.
8. **Test it on a clean machine or VM** (the guide asks for this explicitly).

### B-02 🔴 P0 — The declared model provider is wrong
The README (table row B, "Model/provider"), `results/config.json` (`"model": "gpt-4o"`) and `docs/ARCHITECTURE.md` (lines 19, 45, 63) all say **gpt-4o does tool calling**. The code does not: `cascaded_agent.py` has **no LLM**. It runs Silero VAD → OpenAI whisper-1 → **rule-based `ParticipantAgent`/`nlu.py`** → OpenAI tts-1. The guide requires "a clear declaration of the model provider or custom agent". Declare it as a **custom LiveKit agent** and list every hosted API (OpenAI STT/TTS) and every local model (faster-whisper/CLIP if used).

### B-03 🟠 P1 — Provider name and output file names
The official scripts key everything on `--provider` → `result_{provider}.json`. Pick one name (for example `bilio`), use it in every stage, and put it in the README.

### B-04 🟠 P1 — Versions, seeds and Python are not pinned
- `~=1.3` resolved to **livekit-agents 1.8.3** here. `results/config.json` says 1.3 was the target and 1.8.3 was "tested, not verified". Pin one version.
- `submission.yaml` says Python 3.12 and entry point `agent.agent:ParticipantAgent` (that is the old kit's entry point, not the LiveKit agent). FDB says 3.10. The sandbox has 3.13. Pick 3.11 or 3.12 and enforce it in the script.
- `seeds: null`. Record seeds: TTS is not deterministic, but record the latency-injector seed if one is used, the Whisper `temperature=0`, and so on.
- HF model downloads are not pinned to a revision (`perception.py:130-132`, bandit B615).

---

## 3. LiveKit agent runtime bugs (these affect the benchmark and the live demo)

### B-05 🔴 P0 — The tool executor blocks the whole output pump, so it is not full-duplex
`livekit_agent/adapter.py:180` runs `await self._tool_executor(cid, api, args)` inside `_pump_outputs`. `cascaded_agent.tool_executor` awaits `asyncio.to_thread(registry.call)`, and `MockAPIRegistry` sleeps for the latency profile (normal / slow = **1–3 s** in 9 of the 100 examples). During that time **no filler, no cancel_tool and no speech can be emitted**.
Measured (T10): the interruption arrived at +100 ms, but "Got it — switching to Chicago" and `cancel c1` came out at **+1002 ms**, only after the stale call finished. This breaks the core guideline: "tool calls … run in the background without ever blocking the conversation".
**Fix:** in the pump, run each executor as its own task (`asyncio.create_task`), keep `call_id → task`, and make `tool_canceller` call `task.cancel()`. Mark results of cancelled calls as stale. Add a regression test with an `await asyncio.sleep(1)` executor (the current tests use executors that return immediately, which hides this bug).

### B-06 🔴 P0 — The official runner cannot read the latency telemetry
The official `run_tool_benchmark.py:384-391` reads lines that start with `LATENCY_TRACK_JSON: ` from `/tmp/agent_heartbeat.log`. The local `cascaded_agent.py:118-123` sends `LATENCY_TRACK_JSON` only to `logging.info` and writes only the human-readable report to the file. So `search_latency_breakdown` is never filled. Write the JSON line to the file exactly like the upstream template (upstream lines 101-108). Also write the "JOINING ROOM" heartbeat line to the file as upstream does.

### B-07 🟠 P1 — The latency tracker measures the wrong moments
- `adapter.py:172` computes `kind` and then passes `action` to `speak()`, so a `clarification_request` is never treated as the first response. `kind` is also a dead variable (ruff F841).
- `speak()` stamps `agent_start_at` when `session.say()` is **queued**, not when audio starts. Upstream uses `agent_state_changed → "speaking"`. Use the same event and stamp the first speech of **any** kind, including the filler. The FDB "first response latency" metric rewards a fast filler.
- `tracker.tool_start_at/tool_end_at` are single fields. Chained or concurrent calls overwrite each other, and `tracker.reset()` after the first final response throws away the chain.
- The upstream `log_breakdown` requires `tool_start_at`. Yours does not, so no-tool turns produce a different record.

### B-08 🟠 P1 — Barge-in and partial-transcript hooks do nothing
`attach_livekit_session` registers `user_state_changed`, but the handler body is `pass` (line 222), and `on_user_partial` returns immediately. So there is no early cancel on speech onset, and filler lines already queued with `session.say()` are never cancelled when the user interrupts. Implement this: on user speech onset while the agent is speaking or working, call `session.interrupt()`, drop queued speech handles from older epochs, and (optionally) pre-cancel on a partial that contains a correction marker ("actually", "no wait", "instead").

### B-09 🟠 P1 — Routing a final transcript as an "interruption" is a heuristic
`busy = bool(inflight) or not answered` together with `last_api is not None`. A new, unrelated request after a clarification question counts as a barge-in, and a mid-sentence pause (LiveKit endpointing at 0.5 s) splits one utterance into two turns (T9: 5 of 20 examples split). The second half is then treated as an interruption of the first. **Fix:** buffer finals for a short window (≈700–900 ms, or until the VAD confirms silence) before planning, or plan speculatively and re-plan when a continuation arrives. The FDB recordings are single-turn with long hesitations (mean 47 s, up to 59 s).

### B-10 🟡 P2 — Cancellation cannot stop a thread-bound mock call
`tool_canceller` only adds the call to a set. After B-05 this is fine for read-only calls. For state-modifying calls, keep the ledger reconciliation, but never *issue* a state-modifying call until the user's utterance has settled (see B-09). Otherwise FDB counts the stale call as an **extra tool**, which is an automatic FAIL in `evaluate_pass_rate.py` (it checks precision).

### B-11 🟡 P2 — Cold start and memory
`ParticipantAgent.setup()` loads faster-whisper **and CLIP** (4.3 s warm, ~713 MB RSS) for every session, although the FDB path never uses vision or local ASR (LiveKit does the STT). Load them lazily, or preload once per worker (`prewarm`). The full 100-example audio replay was OOM-killed on this 1 GB sandbox. That is fine on a 48 GB machine, but cold-start cost per room still adds latency.

### B-12 🟡 P2 — The agent's response text doesn't match the official mock schema
16 of 100 transcripts contain the literal string `None` ("FL123 departing None for $None", "Booking reference None"). `agent.py:1314/1346/1361/1371/729` read `depart`, `price_usd` and `booking_id`, but the official mocks return `date`, `price` and `booking_ref`. This hurts the LLM-judge **response quality** score and truthfulness ("no false claims"). Build responses from the actual result keys generically: summarise whatever fields exist and never print a missing value. Also, "Here's what I found for Toronto: drivers_license, DL88" puts a stale subject into an identity update.

---

## 4. Why the benchmark score is low (argument and tool errors)

From the official evaluator on all 100 examples (text mode): **43 wrong-tool failures and 41 wrong-argument failures**.
By domain: finance 40% · ecommerce 10.3% · housing 11.5% · **travel 0%**.
By disfluency: SELF_CORRECTION 5.9% · HESITATION 0% · FILLER 24% · PAUSE 16.7% · FALSE_START 33%.
With state rollback 5.9%, without 18.1%.

| ID | Failure class (examples) | Evidence | Generic fix (no hard-coding) |
|---|---|---|---|
| A-01 | **A spelled ID inside a sentence is not used**, so the agent asks "what order id should I use?" instead of calling a tool (ecommerce_01, _04; 6 × missing `track_order`) | `nlu.spelled_ids("The order ID is A-B-C-1-2-3.")` returns `ABC123`, yet the agent still asks for the ID | Feed `spelled_ids()` into slot filling for every id-typed string argument (`*_id`, `*_number`) before falling back to a question |
| A-02 | **Free-text args swallow the whole utterance** with fillers (ecommerce_02, _09: query = "Like well hmm could you search for running shoes — actually no…") | appendix rows | Extract the noun-phrase span after the trigger verb; strip disfluencies; apply the self-repair rule "last value after a repair marker wins" to **free-text** slots, not only to enums and cities |
| A-03 | **Self-correction keeps the original value** (finance_15 checking→savings; ecommerce_09; housing_17 origin/destination garbled) | SELF_CORRECTION pass rate 5.9% | Run repair resolution on the whole turn *before* slot filling: split on markers (actually / no wait / instead / I mean / scratch that / sorry) and keep the right-most complete clause for the slot being repaired |
| A-04 | **Currency direction reversed** (finance_18: "1000 USD to EUR" → from=EUR) | appendix | Resolve direction from "from X to Y", "X into Y", "convert X … to Y" patterns, with position-based ordering as the fallback |
| A-05 | **`add_to_cart` quantity is not sent** (ecommerce_06, _14; 5 cases). The official `mock_apis.add_to_cart(product_id, quantity)` has **no default**, so a missing quantity **raises TypeError** in the real mock. Your `fdb_tools.py` wrongly marks it optional with default 1. | `fdb_tools.py:127` vs `mock_apis.py` | Fix the manifest (`required: True`). Fill defaults from the schema when the user doesn't say a number. Parse number words ("two", "a couple") |
| A-06 | "Add 2 of product P52" → asks for the product id | probe | The ID regex misses short alphanumerics such as P52 and K2 when they aren't spelled out |
| A-07 | **Numbers bound to the wrong slot**: `bedrooms: 1500` when no bedroom count was said (housing_05) | appendix | Bind a number to a slot only when there is a unit or keyword next to it (bedroom/br, $/dollars/budget/under). If a required number is missing, ask for it or use the schema default |
| A-08 | **Unknown-product search goes to "Happy to help! I can …"** (ecommerce_05 "a desk under 300"); "Search for a desk" asks "which city?" | probe | The tool ranker must consider every tool whose description matches the verb/object (search/find/looking for + product noun). Never ask for a slot the chosen tool doesn't have |
| A-09 | **Multi-step chains stop early** (travel 0%: travel_21 "book it under the name Quinn Davis" → still asks "Whose name?"; conditional chains such as travel_20) | appendix | Extract **every** clause's slots up front (the compound planner exists but doesn't carry `passenger_name` across clauses). Resolve `$RESULT_n` references from earlier results. Handle "if … then … otherwise …" by evaluating the condition on the tool result |
| A-10 | **Extra calls** (for example search_flights issued in a housing scenario; `Unexpected tools` rows) | appendix | Any extra call is an automatic FAIL. Don't guess a tool for a leftover clause with low confidence; ask instead |
| A-11 | Format differences the LLM judge probably forgives (ordinal "August 20th"; `drivers_license` vs `driver_license`) | +12 scenarios if forgiven | Still normalise them (strip ordinals; use the upstream docstring examples as canonical forms, e.g. `driver_license`) so you don't depend on the judge |
| A-12 | Args expected by FDB but missing from your manifest: `search_apartments.pets_allowed` (6 cases), `search_products.category` (1) | schema diff | Your manifest was hand-copied. The official mocks accept `**kwargs`. Treat this as a judgement call and document it: add optional fields only when they come from the tool description, never from test answers |

**Benchmark ethics (disqualification risk) — please read.**
- `agent/nlu.py` has a city gazetteer and lexicons whose values overlap the FDB answers (Atlanta, Austin, Chicago, …, Vancouver; mortgage, savings, checking). A generic gazetteer is fine, but reviewers "check".
- Commit messages such as "FDB-v3 text replay 37→48/100" and "48→57/100" show tuning directly on the public test set.

**What to do:**
- (a) Keep lexicons generic and large, with their source cited (for example, the top 200 world cities).
- (b) Split the 100 examples into a dev half and a held-out half. Tune only on dev and report both.
- (c) Write a short "anti-overfitting statement" in the README.
- (d) Remove anything that matches a benchmark literal and has no general justification.
- (e) Consider an LLM step for argument extraction (a small hosted model is allowed; declare it). This is the standard way to fix A-02/A-03/A-09 without writing rules per test item.

---

## 5. Extension (Triage Line, 20% of the score)

These were probed directly through `TriageCallSession` using the legacy mock STT/TTS/audio (T11):

| ID | Caller says (while a tow dispatch is pending) | Result | Severity |
|---|---|---|---|
| E-01 | "Do not confirm" | 🔴 **dispatched** | P0 safety |
| E-02 | "Yes, but actually I am on Highway 12" | 🔴 dispatched to **Highway 9**, the stale location | P0 safety |
| E-03 | "Yesterday I called" | 🔴 dispatched: `"yes"` matches as a substring of "yesterday" | P0 safety |
| E-04 | The same breakdown report, then "yes", repeated twice | 🔴 **2 dispatches** for one incident (no idempotency key) | P0 (guide: "never perform the same state-changing action twice") |
| E-05 | "…no fire and nobody is injured" | 🔴 "This sounds like an emergency (fire)… escalating", because negation is ignored | P1 |
| E-06 | "Yes" after teardown | 🔴 unhandled `InvalidTransitionError` (aborted → finalized) | P1 |
| E-07 | "No, do not send it" / "Not yet" | ✅ aborted correctly | — |

Root cause: `triage_brain.py:67-78` uses `any(phrase in lowered)` substring matching, and checks affirmative **before** negative.

**Fix:**
- Tokenise the utterance and match words (`\byes\b`).
- Check negation and correction first; a correction wins.
- Treat mixed utterances ("yes but actually…") as a correction: re-deliberate, don't confirm.
- Add an incident-level idempotency key, for example (action_type, normalised location, call).
- Guard confirm after teardown or when the session is closed.
- Detect negated hazards ("no fire").

Missing for the rubric:
- (1) `triage_livekit_agent.py` has **never run in a live room**.
- (2) It isn't in any video.
- (3) It uses no STT for real audio. `stt=None` is passed in, so it relies on LiveKit's STT: document this.
- (4) The guide gives examples: camera-frame troubleshooting, in-car destination change. The project also has a `ui/live.html` PWA with camera and mic. **Pick ONE extension and make it end to end.** The guide says one done well beats three half-built. Right now there are two half-built ones (Triage Line and the live PWA).

---

## 6. Submission checklist vs the guide

| Guide requirement | Status | Action |
|---|---|---|
| Code repo + README with an architecture diagram, exact setup/run steps, extension clearly marked | 🟠 README exists; the diagram is in `docs/ARCHITECTURE.md` (mermaid) but is wrong (gpt-4o) | Rewrite the README: one correct diagram, one "Quick start", "Reproduce FDB-v3", "Extension", "API keys", "Models used". Move history docs (P1/P2/LIVEKIT_* / README_PREVIOUS / KIT_README) to `docs/history/` |
| One-command reproduction script (install, configure, evaluate) + provider declaration | 🔴 Stops by design (B-01, B-02) | §2 |
| Benchmark results + run logs (scores, seeds, config) from your own best run | 🔴 `results/results.md` = "NOT RUN" | Commit the real reports from a live run; mark offline replay numbers as such |
| Demo video, 3–5 min: a real interruption on the benchmark + the extension | 🔴 Missing (only `docs/VIDEO_SHOTLIST.md`) | Record a single take: an FDB example with self-correction running live in LiveKit (show the agent log with cancel_tool), then the extension live |
| Slide deck ≤ 8 slides: problem, architecture, results, next steps | 🔴 Missing (only `docs/DECK_OUTLINE.md`) | Build the deck from real numbers |
| API keys documented, not included | ✅ `.env.example`, `.gitignore` covers `.env.local` | Also document the judge key (OPENAI_API_KEY with gpt-4o access) |
| Pin seeds and versions | 🔴 | B-04 |
| No calls to your own servers at evaluation time | ✅ Only OpenAI and LiveKit | — |
| No cross-scenario caching | ✅ New `TriageAdapter` per room. ⚠️ The module-level `registry` and `/tmp` logs are shared (as upstream does) | Keep it that way; don't add caches |
| No hard-coding of test items | ⚠️ Risk (§4 ethics box) | Dev/held-out split + statement |
| Old-kit PDF items: python 3.10–3.12, 120 s/scenario, 300 s warm-up | ⚠️ The sandbox is 3.13; setup loads CLIP needlessly | B-04, B-11 |

---

## 7. Code quality and security (low priority)

- ruff: 20 × blind `except Exception`, 9 unused imports, 5 unused variables (for example `adapter.py:172 kind`), 4 × B904, 2 × blocking `open()` in async code (`cascaded_agent.py` telemetry → use `asyncio.to_thread` or a logger handler).
- bandit (medium): HF downloads without a pinned revision (`perception.py:130-132`); fixed `/tmp` paths (these match upstream, keep them but document); UI binds 0.0.0.0 with CORS `*` and no auth (fine for a demo; add a token if exposed publicly).
- `ui/live.py`: uploads are saved under `live_uploads/<sid>/` without MIME validation (a "data:image/png;base64,AAAA" request was accepted as `.jpg`). Validate with PIL before saving, and delete the files when the session ends.
- `livekit_agent/cascaded_agent.py` docstring says the adapter and agent are "UNMODIFIED", but both have been modified many times since. Keep comments truthful, because reviewers read them.
- Stale claims to correct: README "stops at Stage 1 (no PyPI egress)" (false: pip works); "191/191" (now 209); readiness score "52/100" (self-assessed, not reproducible).

---

## 8. Prioritised to-do list (ready to hand to an AI)

**P0 — without these the benchmark share is ~0 and the extension is unsafe**
1. Rewrite `run_fdb_v3.sh` stages 1–6 exactly as in B-01, and test it on a clean VM with `--limit 5`, then the full run.
2. Adapter: tool calls run as tasks, real cancellation, a slow-executor regression test (B-05).
3. Telemetry: write `LATENCY_TRACK_JSON:` to `/tmp/agent_heartbeat.log`; stamp `agent_state_changed=speaking`; track per call (B-06, B-07).
4. Correct the provider declaration everywhere (B-02).
5. Triage: word-level yes/no, correction precedence, incident idempotency, post-teardown guard (E-01..E-06), plus a test for each.
6. `add_to_cart.quantity` required + default handling (A-05).

**P1 — raises the benchmark score**

7. Settle the utterance before planning; speculative planning with re-plan (B-09, B-10).
8. Argument extraction: spelled IDs in context, free-text span extraction, self-repair for all slot types, currency direction, unit-anchored numbers, chain slot carry-over, `$RESULT` binding, conditional chains (A-01..A-09). Consider a declared small LLM for extraction.
9. Response text from the real result keys; never say "None" (B-12).
10. Barge-in: `session.interrupt()` on user speech onset and dropping stale queued speech (B-08).
11. Dev/held-out split of the 100 FDB examples; report both; add the anti-overfitting statement.
12. Pin Python, livekit-agents, the FDB commit and HF revisions; record seeds (B-04).

**P2 — docs, video, polish**

13. Record the 3–5 min video and make the ≤8-slide deck; rewrite the README (§6).
14. Choose one extension and run it live in LiveKit; show it in the video.
15. Lazy model loading / worker prewarm (B-11); lint/bandit cleanup (§7).

**Target after the fixes:** re-run the text replay (a quick proxy), then the live run. Aim for a text strict pass above 50% before spending time on the live run. The published cascaded/realtime baselines are listed in the FDB-v3 paper (arXiv 2604.04847). Compare against those in the deck.

---

## 9. How to reproduce this audit

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "livekit-agents[openai]~=1.3" livekit-plugins-silero "livekit[crypto]~=1.0" python-dotenv numpy pytest fastapi httpx faster-whisper gdown ruff bandit
git clone https://github.com/DanielLin94144/Full-Duplex-Bench /tmp/fdb   # audited at 3e799c45
mkdir -p livekit_agent/.fdb_v3_repo && cp -r /tmp/fdb/v3 livekit_agent/.fdb_v3_repo/
gdown 1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz -O fdb.zip && unzip -q fdb.zip -d data
pytest tests -q; (cd legacy && pytest tests -q); for f in livekit_agent/adapter_tests/test_*.py; do python $f; done
python run_local.py --all --agent agent.agent:ParticipantAgent
python livekit_agent/fdb_v3_offline_replay.py --data data/fdb_v3_data_released --text
cd livekit_agent/.fdb_v3_repo/v3 && python evaluate_pass_rate.py --benchmark benchmark_data_v2.json \
   --results-dir ../../../data/fdb_v3_data_released --provider bilio_text --output pass.json   # add --use-llm with a gpt-4o key
```
