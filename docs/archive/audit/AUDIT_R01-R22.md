BILIO: guideline alignment, bug audit, and implementation plan
Bottom line: your project is a useful prototype for the older Theme 5 kit, but it is not yet submission-ready under the updated participant guide. There are also significant correctness and safety bugs that the existing tests do not catch.

I read the uploaded conversation, both guides, and the current implementation, then reran tests and targeted reproductions.

No source-code fixes were applied. No tracked files were changed.

1. What happened last time—and what I checked now
The previous conversation had already:

Identified the mismatch between the two guides.
Reviewed the active agent and live UI.
Passed the 24 regression tests.
Obtained approximately 89.1/100 on the older public suite in this sandbox.
Reproduced several serious bugs.
However, the uploaded conversation ends without the actual final report, even though its task list says the report was completed.

The current checkout is commit f33099b. Its history includes the earlier commit named:

fix(agent,ui): implement audit fixes B01-B38, regression tests, live mobile assistant PWA

That implementation is present. A commit saying “fixes B01–B38” does not mean all related failure modes are fixed. Several tests cover only a narrow successful example.

This report uses new identifiers R01–R22 so you do not confuse these remaining findings with the older B-numbered issues.

Evidence labels
Reproduced: demonstrated using the current code, with controlled inputs or injected tool/perception results.
Code inspection: established from the implementation, but not validated in a real browser/provider deployment.
Missing/unverified: not present in the inspected submission, or not demonstrated by available evidence.
This is a thorough audit of the active path, not a guarantee that no other bugs exist. I did not run real LiveKit inference, real paid APIs, or a physical-device microphone test.

2. The biggest issue: the two guides specify different submissions
Which guide your project follows
Area	Older PDF / bundled kit	Updated participant DOCX	Current project
Agent interface	Two asynchronous queues	LiveKit voice agent	Queue-based
Benchmark	Nine public scenarios and hidden kit scenarios	Full-Duplex-Bench v3	Older local harness
Tool environment	Flight search, booking, manuals, tickets, manifest tools	12 tools across four domains	Older tools
Scoring	Task/recovery/latency/safety	FDB-v3 metrics with LLM judge	Older deterministic scorer
Reproduction	Import agent and replay scenarios	End-to-end FDB-v3 reproduction script	run_local.py only
Extension	Multimodal examples within kit	Working use case beyond benchmark	Mock-backed device/manual demo
Submission assets	Kit package	Code, reproduction script, results/logs, video, slides	Incomplete evidence
Recommendation: treat the updated DOCX as the working submission target, and ask the organizers to confirm that it supersedes the PDF. Do not spend the remaining time optimizing the old scoring system while ignoring the updated one.

I also checked the upstream FDB-v3 README, which confirms the LiveKit inference workflow, tools, and evaluation commands.

Why this matters to your score
The updated guide allocates:

60%: benchmark performance on the organizers’ rerun.
20%: working extension.
20%: documentation, architecture, and video.
It explicitly warns that an unreproducible benchmark submission can receive zero for the benchmark portion after the specified follow-up opportunity.

Your current local scores cannot be converted into an FDB-v3 score.

3. What is already implemented well
There is useful engineering here. You do not need to discard everything.

Working foundations
Non-blocking event consumption

ASR and vision run outside the main event-consumer path.
Controlled slow-perception tests confirm that explicit interruptions can still be processed.
Tool-call tracking

Calls have identifiers.
The agent maintains an in-flight ledger.
Targeted cancellation works for several tested flight corrections.
Some stale-result protection

Cancelled call results are excluded from normal response grounding.
Dependency checks prevent certain outdated flight-search results from being reused.
Basic operation deduplication

Repeated identical successful state-modifying calls are suppressed.
Failed operations can be retried.
Basic clarification

Missing names and enum values can be requested.
A bare “yes” does not select between two offered cities.
Perception fallback

Missing models generally cause clarification rather than a crash.
Useful debugging interface

Scenario replay, trace display, snapshots, and comparison views.
The live interface explicitly labels its tools as simulated.
Existing regression suite

A good starting point, but it needs substantially broader coverage.
Overall: the architecture demonstrates the right ideas, but its state ownership, authorization, and live-audio integration are not yet robust.

4. Fresh validation results
Existing regression tests
Executed:

CopyPYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/
Result: 24 passed.

Older public suite
Executed at time scale 1.0, using the standard 6000 ms tail:

Scenario	Score
pub_01: simple text search	100.0
pub_02: text interruption	100.0
pub_03: chained booking	100.0
pub_04: no-tool conversation	100.0
pub_05: audio ambiguity	53.8
pub_06: audio disfluency	56.9
pub_07: visual lookup	90.8
pub_08: tool failure/retry	100.0
pub_09: unseen tool	100.0
Arithmetic mean	89.1
All four scenarios_extra/ scenarios scored 100.0.

No agent_crash, protocol_error, or tool_abandoned entries appeared in those runs.

Important limitations
This environment lacks:

faster_whisper
huggingface_hub
tokenizers
livekit
OCR is available; the full Whisper/CLIP stack is not.

Therefore:

The audio results measure fallback behavior.
The visual result does not validate full CLIP operation.
The scores exclude the updated FDB-v3 evaluation.
These runs do not establish actual microphone-to-audible-response latency.
Passing these scenarios does not establish safety. The additional probes below demonstrate why.

5. Confirmed remaining bugs and proposed fixes
Priority definitions
P0: safety-critical or submission-blocking.
P1: major functional failure.
P2: robustness, quality, or operational issue.
The P0 side-effect risks are currently contained by mock tools. They become much more serious if you attach real booking, payment, ticketing, or account-management services.

R01 — Revoking booking permission does not stop the booking plan
Priority: P0
Evidence: Reproduced
Location: agent/agent.py:622–680, 757–791

Reproduction
“Find flights to Boston and book for Alice.”
While search is pending: “Actually don’t book, only show options.”
Return the search results.
Observed: the agent still emits book_flight.

Cause
revise() considers only certain slot changes. When consent changes but destination/name/time do not, it returns “Okay — still on it.”

Additionally:

Copyplan = self.plan or c.get("plan") or []
can restore an old booking plan from a call record after the current plan has been cleared.

Fix specification
Represent permission to perform a side effect explicitly, independently of slots.
Process revocation before slot-change detection.
Invalidate dependent commit steps.
Treat an explicitly empty current plan as authoritative.
Recheck current authorization immediately before issuing a state-modifying call.
Acceptance: after permission is withdrawn, late search results may show options but must never trigger booking.

R02 — Negated non-booking actions are executed
Priority: P0
Evidence: Reproduced
Location: agent/nlu.py:54–55,223–241; agent/agent.py:336–387,499–505

Reproductions
“Do not open a support ticket for my broken TV.”
Emits create_support_ticket.
“Do not cancel booking BK-0001.”
Emits cancel_booking.
Cause
Negation handling is specialized for booking. Keyword-based routing still selects other forbidden actions.

Fix specification
Extract action polarity for every requested operation.
Distinguish “cancel my booking” from “don’t cancel my booking.”
Gate all state-modifying calls on positive authorization.
If scope is ambiguous, clarify instead of executing.
Acceptance: negated state-changing requests produce zero forbidden tool calls, including for dynamically supplied tools.

R03 — Cancellation is treated as proof that a side effect did not happen
Priority: P0
Evidence: Reproduced with an injected late result
Location: agent/agent.py:232–240,646–664,692–696; ui/live.py:101–108

Reproduction
Start booking Boston.
Interrupt: “Actually make it Seattle.”
Deliver a late successful Boston booking result.
Deliver Seattle search results.
Observed:

“I stopped the earlier booking before replacing it.”
The Boston success is ignored.
A Seattle booking is issued.
The original operation remains unknown.
Why dangerous
Cancelling a local task is not proof that a remote service rolled back its operation.

Fix specification
Keep a durable-in-session operation record after cancellation.
Separate:
cancellation requested;
cancellation confirmed;
committed;
outcome unknown.
Consume late results for reconciliation, even when they cannot drive the old conversational plan.
Block conflicting replacement commits until the old outcome is resolved.
Use truthful language: “I requested cancellation” when confirmation is unavailable.
Acceptance: a late commit is recorded and disclosed; a replacement cannot silently create two active bookings.

R04 — An unknown outcome becomes retryable merely because the user repeats the request
Priority: P0 before real providers
Evidence: Reproduced
Location: agent/agent.py:205–221,706–709

Observed
For a cancelled live rental reservation:

Operation becomes unknown.
First repeated request changes it to failed.
Second repeated request issues the operation again.
There is no evidence that the first attempt failed.

Fix specification
Never turn unknown into failed because a warning was spoken.
Distinguish definitive rejection from transport timeout/unknown commit status.
Reconcile using provider status or a stable idempotency key.
An explicit retry decision must not discard the original operation identity.
Acceptance: repeated utterances cannot bypass unknown-outcome protection or create another provider-side commit.

R05 — Rebooking after a confirmed cancellation returns a stale “already booked” answer
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:205–215,737–749

Reproduction
Book Boston for Alice.
Successfully cancel that booking.
Ask to book the same flight for Alice again.
Observed: the agent says the original booking is already done and returns the cancelled booking reference.

Cause
The booking operation remains permanently successful in the deduplication ledger; cancellation does not update its business lifecycle.

Fix specification
Separate request deduplication from reservation status.
Link cancellation to the affected reservation/operation.
After confirmed cancellation, allow a newly authorized booking operation.
Acceptance: one new booking is created after confirmed cancellation, and the cancelled reference is never described as active.

R06 — Clarification answers are overwritten by the original utterance
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:393–450,466–482,775–777

Reproductions
ASR proposes passenger “Alice”; the user answers “Priya.”
Booking still uses Alice.
Requested departure 14:45 is unavailable; user selects “2 PM.”
Agent searches again and asks about 14:45 again.
Cause
The clarification updates a slot, then the original transcript is routed/parsing runs again and restores the old value.

Fix specification
Apply clarification as a typed update to the existing task.
Give that confirmed update precedence over older transcript-derived values.
Normalize times through the time parser.
Resume the relevant plan step without reinterpreting the original request as new input.
Acceptance: “Priya” reaches the booking arguments; “2 PM” selects 14:00 without looping.

R07 — Partial text buffered before a correction restores stale intent
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:126–131,622–680

Reproduction
Partial chunk: “Book a flight to Boston for Alice.”
Interruption: “Actually make it Seattle.”
Final chunk: “please.”
Observed: the next search is for Boston.

Fix specification
Associate buffered chunks with utterance/task versions.
Retire or explicitly reconcile the pre-interruption buffer.
Preserve the structured correction rather than reassembling stale text blindly.
Acceptance: trailing pre-correction fragments cannot restore Boston.

R08 — Old ASR can become a new request after a newer user turn
Priority: P1
Evidence: Reproduced with delayed ASR
Location: agent/agent.py:132–139,150–155,244–253,336–360

Reproduction
Begin slow transcription of a flight-booking request.
Submit a newer complete support-ticket request.
Let the older transcription finish.
Observed: the old flight request executes afterward and changes the active state.

Cause
ASR completion checks the global version, but ordinary newer turns do not consistently advance/invalidate the relevant work.

Fix specification
Assign input sequence and task ownership when audio arrives.
Make supersession rules apply to ordinary turns as well as explicit interruption events.
Reject or reconcile out-of-order completions according to their task identity.
Acceptance: obsolete ASR cannot reactivate a superseded request.

R09 — Corrections to origin and arbitrary schema fields are ignored
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:617–645,527,680

Reproductions
“From Seattle to Boston” → “Actually from Chicago.”
Origin remains Seattle.
Rental reservation for compact → “Actually make it SUV.”
Compact reservation is not cancelled/replanned.
Cause
Revision supports only destination, date, passenger, and departure time. Dependency tracking is similarly restricted.

Fix specification
Build typed corrections from the active tool schema.
Support nested fields, enums, origin, booleans, numbers, and identifiers.
Track actual argument dependencies.
For already-running state changes, use the reconciliation rules from R03.
Acceptance: corrected arguments are applied across declared schema fields, and only affected work is invalidated.

R10 — Lowercase passenger names can silently reuse an earlier passenger
Priority: P1
Evidence: Reproduced
Location: agent/nlu.py:51,165–177; agent/agent.py:466–479

Reproduction
Complete a booking for Alice.
“Please find and book a flight to Seattle for bob smith.”
Observed: Seattle is booked for Alice.

Cause
Inline name extraction expects capitalization. When extraction fails, the old session slot remains usable.

Fix specification
Support role-aware lowercase and multiword names.
Treat explicit but unresolved passenger information as unresolved—not permission to reuse an old name.
Store passenger slots under the relevant task.
Acceptance: the request uses Bob Smith or asks for confirmation; it never silently uses Alice.

R11 — A new booking request is mistaken for a short search follow-up
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:362–380

Reproduction
After a completed Boston search:

“Book a flight to Seattle for Alice.”

Observed: Seattle is searched, but no booking follows.

Cause
The elliptical-follow-up shortcut runs before booking intent is resolved. Its stopword-filtered token count makes a complete request look short.

Fix specification
Resolve explicit action changes before follow-up shortcuts.
Restrict ellipsis to genuine slot-only answers.
Do not use token count alone to determine whether a request is complete.
Acceptance: explicit booking intent survives a preceding search-only task.

R12 — Flight-selection constraints are ignored
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:757–791

Reproduction
“Find and book the cheapest flight to Boston for Alice.”

Given:

08:00 flight: $129.
14:00 flight: $99.
Observed: the $129 flight is booked.

“Book the second one” also fails to bind directly to the previously presented second option.

Fix specification
Preserve selection criteria and the presented candidate set.
Resolve cheapest, earliest, ordinal selection, explicit ID, and time against actual results.
Clarify unsupported/conflicting criteria before committing.
Acceptance: cheapest selects $99; “second one” selects the displayed second candidate without silently defaulting to the first.

R13 — Known and unknown city extraction do not work together
Priority: P1
Evidence: Reproduced
Location: agent/nlu.py:99–136

Observed
Input	Incorrect result
“Flights from Boston to Kochi”	Destination missing
“Flights from Kochi to Boston”	Origin missing
“Find flights to san jose”	Destination becomes San
Cause
Fallback extraction only runs when no known city is found anywhere. Lowercase fallback captures one word.

Fix specification
Extract origin and destination spans independently.
Combine gazetteer matches with contextual span extraction.
Preserve multiword place names.
Clarify when ambiguous rather than truncating.
Acceptance: all three cases preserve the full origin/destination correctly.

R14 — Enum self-correction picks schema order instead of the corrected value
Priority: P1
Evidence: Reproduced
Location: agent/nlu.py:270–274,355–358

Reproduction
“Reserve compact, actually SUV in Denver.”

Observed: car_class="compact".

Fix specification
Find enum mentions with positions.
Apply correction scope and negation.
Select the authoritative mention, not the first enum in the schema.
Acceptance: SUV is selected regardless of enum declaration order.

R15 — Argument construction does not enforce schema constraints
Priority: P1
Evidence: Reproduced, plus code inspection
Location: agent/nlu.py:313–339,389–414; agent/agent.py:197–229

Reproduction
For required integer nights with minimum: 1:

“0.5 nights”

Observed: arguments contain nights: 0 and no missing/error indication.

Additional gaps
No comprehensive pre-call validation of bounds, formats, array items, or standard JSON Schema semantics.

Fix specification
Separate extraction from validation.
Normalize each adapter’s schema into one internal representation.
Reject fractional values for integer fields rather than truncating.
Validate the complete arguments immediately before dispatch.
Clarify invalid fields.
Acceptance: invalid values never reach the tool; valid False and zero remain supported where the schema permits them.

R16 — Malformed “success” results produce false completion claims
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:692–751

Reproduction
Return a nominal booking success without a booking ID.

Observed:

“Done — you’re booked … Booking reference None.”

Fix specification
Validate status and result structure per tool.
Require reliable completion evidence for state changes.
Treat incomplete success as malformed/unknown, not safely retryable failure.
Never invent IDs or completion facts.
Acceptance: malformed booking/ticket/cancellation responses do not produce “done” claims or automatic duplicate attempts.

R17 — A newer camera frame can still produce a lookup grounded in an old frame
Priority: P1
Evidence: Reproduced with delayed vision
Location: agent/agent.py:156–163,258–282,544–586

Reproduction
Old frame analyzed as HDMI.
Middle frame processing.
User asks about the port.
New frame arrives, representing Ethernet.
Observed: lookup query uses the old HDMI label.

Cause
A completion can release a waiting request even though its result is not installed as current. _issue_manual() then reads an older self.vision.

Fix specification
Bind every visual result to its frame ID.
Pass the qualifying result directly to the lookup.
Define whether a question targets the frame at question time or the latest frame.
Never satisfy a wait using an unrelated cached result.
Acceptance: the lookup uses its declared target frame or explicitly reports uncertainty.

R18 — Live server ASR bypasses the agent’s uncertainty handling
Priority: P1
Evidence: Reproduced with an injected transcript
Location: ui/live.py:160–173; agent/agent.py:296–333

Reproduction
ASR returns:

“book flights to Boston for Alice”;
confidence 0.01;
alternate transcription naming Austin.
Observed: the live adapter passes the transcript straight into user_text().

Word confidence and alternatives do not reach the normal audio clarification logic.

Fix specification
Use one perception-result contract for live and harness paths.
Preserve confidence, alternatives, utterance identity, and timing.
Apply the same uncertainty gate before side effects.
Handle stale live-ASR completions using R08’s ordering rules.
Acceptance: conflicting Boston/Austin interpretations trigger clarification instead of booking.

R19 — Answering a clarification while the assistant is speaking loses the answer
Priority: P1
Evidence: Reproduced through the live adapter
Location: ui/live.py:132–140; agent/agent.py:595–639

Reproduction
Assistant asks whose name to book under.
While it is speaking, user answers “priya sharma.”
Observed:

Adapter classifies the answer as interruption.
Agent says “Okay — still on it.”
Clarification remains unresolved.
No booking occurs.
Fix specification
Separate:

physical interruption of speech playback;
semantic processing of the user’s utterance.
A barge-in may still be a valid clarification answer.

Acceptance: clarification answers work identically whether they arrive during speech or after speech finishes.

R20 — Tool-provider exceptions strand live operations
Priority: P1
Evidence: Reproduced with injected ConnectionError
Location: ui/live.py:112–122

Observed
Exception escapes _exec().
Pending entry remains.
No result/error event reaches the agent.
Fix specification
Catch provider exceptions and timeouts.
Clean bookkeeping in finally.
Emit structured terminal/unknown-outcome events.
Distinguish read-only retryable failures from uncertain side effects.
Keep user-facing errors useful without exposing secrets.
Acceptance: every operation reaches a known lifecycle state, and the interface cannot remain “running” forever after an exception.

R21 — Filler deduplication causes silence on later legitimate turns
Priority: P2, with responsiveness impact
Evidence: Reproduced
Location: agent/agent.py:176–189

Reproduction
In live mode:

Search Boston and receive the answer.
Request the same search again.
Observed: a tool call is emitted without an immediate spoken acknowledgement.

Cause
Exact-text suppression is session-wide, even though live mode otherwise has a per-turn budget.

Fix specification
Scope live acknowledgement deduplication to a turn or bounded recent window.
Maintain separate old-benchmark filler-budget rules where necessary.
Test responsiveness per user turn, not just the first few turns.
Acceptance: repeated legitimate requests still receive timely, non-spammy feedback.

R22 — Concurrent tasks contaminate each other’s completion messages
Priority: P1
Evidence: Reproduced
Location: agent/agent.py:358–360,730–747

Reproduction
Boston booking is pending.
User asks for weather in Seattle.
Boston booking completes.
Observed:

“Done — you’re booked on FL-BOS-8AM to Seattle for Alice.”

Cause
The completion uses shared current slots rather than the booking’s captured context and result.

Fix specification
Give each task its own state and result context.
Ground completion messages in the originating operation.
Merge results into session memory deliberately.
Do not label an old operation with another task’s location, person, or device.
Acceptance: the booking remains Boston and the weather request remains Seattle, regardless of completion order.

6. Additional live-system and operational concerns
These should also become work items, but they are not all equivalent to the reproduced core bugs.

Finding	Evidence and location	Required change
Server learns about barge-in too late	Code inspection: ui/static/live.js:121–160. Playback stops locally, but semantic input reaches the server after recognition/recording completes.	Send speech-start/turn events promptly; suspend unsafe commits during unresolved interruption. Validate with real audio timing.
Current UI does not demonstrate continuous full-duplex voice operation	Browser recognition is non-continuous; server fallback is record-then-upload. Hands-free restart occurs after speech output.	Implement the LiveKit audio path with VAD, interruption handling, streaming output, and measured cut-off latency.
Queued server speech has no task/version tag	ui/live.py:83–110; ui/static/live.js:49–56,83–104. Clearing local TTS does not identify stale later-arriving SSE speech.	Attach task/turn/output versions; reject obsolete speech and guard TTS completion callbacks.
SSE can lose events	ui/live.py:83–90; ui/server.py:136–160. Queue overflow silently drops events; no replay IDs; multiple consumers share a queue.	Sequence events, support replay/resync, and define single-client or fan-out behavior.
Session cleanup is incomplete	ui/live.py:176–208. Reaping happens when a new session starts; uploaded media is not removed; shutdown cancels tasks and immediately stops the loop.	Periodic reaper, graceful awaited shutdown, media retention limits, and thread/loop cleanup.
Unbounded live ASR workload	ui/live.py:160–173. Each upload starts a thread; global ASR lock serializes decoding.	Bounded queue, per-session limits, stale-job discard, and backpressure. Cancelling to_thread does not stop its running native work.
Malformed HTTP input handling	Reproduced: ui/server.py:74–89,114–121. Non-object payload causes AttributeError; negative content length reaches read(-1).	Reject invalid types and lengths with 400/413, enforce read deadlines, and validate finite numeric values and manifest limits.
Public exposure needs additional controls	Code inspection: session count limits exist, but session creation, uploads and SSE do not have comprehensive rate/resource controls.	Keep private during development; add rate limits, upload quotas, connection limits, and appropriate deployment access controls before public use.
Recorder timeout may stop a later recording	Code inspection: ui/static/live.js:160 closes over mutable global mediaRec.	Capture the recorder instance and clear its timer on stop.
Session recovery is incomplete	ui/static/live.js:30–43,109–115,194. Stored SID is not restored at startup; expired-session sends are not replayed after restart; new-session cleanup does not explicitly stop recording.	Implement explicit lifecycle states and preserve unsent input with user-visible recovery.
Model initialization concerns
agent/perception.py:45–73,122–140 also deserves hardening:

Model initialization is not protected by a dedicated initialization lock.
Failed initialization can become a permanent False sentinel for that process.
Exceptions are mostly swallowed.
Concurrent live requests/preloading can race with initialization.
Model revisions are not pinned.
Use explicit uninitialized/loading/ready/failed states, useful diagnostics, bounded initialization, and controlled retry.

Do not assume model-weight reuse equals prohibited conversation caching. Separate immutable model resources from session data, and confirm the organizers’ interpretation of the guide’s no-cross-scenario-caching rule.

7. What is missing for the updated submission
A. LiveKit integration — missing
The active project contains no working LiveKit agent entry point.

The legacy/providers/live/ package does not supply one either.

Required implementation
LiveKit session/room lifecycle.
Input audio handling and turn detection.
Actual speech output.
Interruption-aware playback.
Tool event/result logging compatible with FDB-v3.
Fresh conversation state for each benchmark example.
Documented credentials and configuration.
A browser PWA is not a substitute for the required LiveKit wrapper.

B. FDB-v3 tool and planning compatibility — missing
FDB-v3 uses:

Domain	Tools
Travel/identity	search_flights, book_flight, update_identity_doc
Finance/billing	get_card_benefits, get_exchange_rate, modify_autopay
Housing/location	search_apartments, calculate_commute, update_search_filter
E-commerce	track_order, search_products, add_to_cart
Your current flight_search → book_flight logic is not a general implementation of these workflows.

Required implementation
Inspect and normalize the actual benchmark tool contracts.
Support one-, two-, and three-step plans.
Bind later arguments to earlier results.
Handle corrections to pending dependent steps.
Validate arguments before execution.
Avoid extra tool calls, because FDB-v3 precision and strict pass rate penalize them.
Simply renaming flight_search to search_flights will not solve this.

A structured slow-path planner is a reasonable approach. It may use an LLM or another implementation, but the guide does not require a particular architecture.

C. One-command reproduction — missing
run_local.py runs the older kit. It does not install, configure, run LiveKit inference, and evaluate FDB-v3.

Required script behavior
One documented command should:

Verify a supported environment.
Install pinned dependencies.
Obtain or verify the pinned benchmark revision and dataset.
Check required credentials without printing secrets.
Start the declared agent.
Run inference.
Run tool accuracy and strict-pass evaluation with the LLM judge enabled.
Run latency analysis.
Save logs, configuration, versions, and results.
Shut down cleanly and return a nonzero exit status on failure.
Do not silently reuse old result files as if they were a fresh run.

D. Reproducibility and honest reporting — incomplete
Specific issues
requirements.txt uses broad or unpinned versions.
Model revisions are not pinned.
CPU/GPU defaults select different ASR models.
No FDB-v3 results were found in the active submission.
README still says media is absent, although media files now exist.
README’s “100” results require clearer provenance.
Console references to “official” timing apply to the old kit, not the updated benchmark.
docs/SUBMISSION.md references eval_submission.py, which is not present in this checkout.
Required changes
Pin dependencies, model revisions, and benchmark revision.
Declare the exact evaluated provider/model.
Publish hardware and runtime configuration.
Label old-kit results as old-kit results.
Document whether bundled media is original kit media or synthetic practice material.
Preserve official benchmark/harness code rather than adjusting it to improve scores.
E. Working extension — partial
Device troubleshooting with a camera is a sensible extension beyond FDB-v3.

However, the current implementation uses:

A small mock manual corpus.
Simulated ticket creation.
Simplified visual labels.
Mock “hybrid” retrieval that boosts keyword hits when an embedding is present rather than performing real vector retrieval.
That is useful for testing, but it does not yet establish a real end-to-end troubleshooting system.

Recommended extension
Implement one focused camera/manual assistant:

Accept a real frame.
Identify or clarify the device/connector.
Retrieve from actual included, appropriately usable manual documents.
Explain using retrieved content.
Cite document/page evidence.
Handle a correction or a replacement frame.
Complete the demonstration without hidden manual intervention.
You do not need to connect real commercial booking/payment systems merely to make the extension credible. A genuine read-only troubleshooting workflow is safer and more focused.

F. Submission presentation assets — unverified/missing from inspected material
The updated guide requires:

A 3–5 minute video:
real benchmark interruption;
extension running end to end.
A deck of at most 8 slides.
Exact setup/run instructions.
Architecture explanation.
Best-run benchmark results and logs.
Provider/key setup instructions without secret values.
I did not find these completed submission assets in the inspected repository. They may exist elsewhere, but they are not demonstrated here.

8. Recommended repair architecture
The recurring failures are not just isolated regex mistakes. They come from shared mutable state without sufficiently explicit task and authorization boundaries.

Recommended internal separation
CopySession
 ├─ ordered input / speech events
 ├─ task registry
 │   ├─ task identity and version
 │   ├─ typed slots with provenance
 │   ├─ current plan
 │   ├─ authorization state
 │   └─ pending clarification
 ├─ operation ledger
 │   ├─ provider idempotency key
 │   ├─ dispatched arguments and context
 │   ├─ cancellation/reconciliation state
 │   └─ validated result
 └─ versioned speech output
Rules the implementation should enforce
A clarification modifies the existing task; old text cannot undo it.
A cancelled plan cannot be revived by a late result.
Permission to act is distinct from having enough arguments.
Cancelling computation does not prove cancellation of a side effect.
Completion messages use the originating operation’s context.
Every perception result belongs to an identified utterance/frame.
Physical speech interruption and semantic intent are separate.
No state-changing call is issued without current authorization and validated arguments.
Keep the fast path small. The slow path can propose plans and interpretations, but a deterministic coordinator should enforce these rules.

9. Implementation order
Phase 1 — Confirm the submission target
Obtain organizer confirmation that the updated DOCX controls.
Pin the FDB-v3 revision.
Mark old-kit documentation and scores clearly.
Exit condition: the team knows exactly what will be evaluated.

Phase 2 — Lock down side-effect safety
Address:

R01–R05.
R16.
R22.
Task-scoped authorization and operation reconciliation.
Exit condition: withdrawal, timeout, cancellation, and late-result tests cannot create unauthorized/conflicting commits.

Phase 3 — Repair state and conversation handling
Address:

R06–R15.
R17.
R21.
Exit condition: corrections and clarifications remain authoritative across text, audio, and tool completion order.

Phase 4 — Unify live audio and benchmark integration
Address:

LiveKit wrapper.
R18–R20.
Speech-start handling and versioned output.
Actual full-duplex timing.
Exit condition: the same coordinator works with live audio and benchmark audio, without bypassing confidence or authorization checks.

Phase 5 — FDB-v3 planning and evaluation
Implement all relevant tool schemas and dependent chains.
Run all benchmark examples.
Evaluate with the LLM judge enabled.
Analyze failures by domain, chain length, and disfluency.
Exit condition: a clean environment can reproduce the run and its reports.

Phase 6 — Extension and submission package
Finish one real extension.
Record the video.
Create the deck.
Update documentation and results.
Perform a clean-machine rehearsal.
Avoid spending significant time on UI polish before Phases 2–5 are complete.

10. Required test expansion
The 24 existing tests should remain, but they are insufficient.

Add at least these test groups:

Test group	Must cover
Authorization	Negation, withdrawal during search, withdrawal during commit, negated cancellation
Operation lifecycle	Late success after cancellation, unknown outcome, timeout after possible commit, legitimate rebooking
Clarification	Different passenger, corrected time, multi-option “yes,” answer during TTS
Input ordering	Old ASR finishes late, multiple audio turns finish out of order, stale partial text
Schema handling	Nested corrections, enums, origins, bounds, fractional integers, array validation
Selection	Cheapest, earliest, second option, explicit IDs, unavailable criteria
Parallel tasks	Different cities/passengers/devices; responses remain bound to their own operations
Vision	Old/middle/new frame races, timeouts, low confidence, target-frame consistency
Provider failures	Exceptions, malformed success, timeout, disconnect, pending cleanup
Live transport	SSE reconnect/overflow, stale speech, session expiry, recording/new-session races
Real voice latency	User speech end → first audible response; speech start → playback stop
Isolation	Fresh state per benchmark example; no reuse of prior conversation answers
Use deterministic event barriers and controlled fake providers for race tests rather than relying only on arbitrary sleeps.

The safety invariants should also be tested across many event orderings, not just one hand-written sequence.

11. Copy-paste handoff for a future coding AI
Use this with the full report:

Review BILIO at commit f33099b and implement the attached audit in controlled phases. The current system targets the older queue-based Theme 5 kit; the updated participant guide requires a LiveKit voice agent evaluated on Full-Duplex-Bench v3.

Before implementation, inspect the actual files and official benchmark contracts. Treat R01–R22 as independently reproducible findings, not as assumptions that the old B01–B38 fixes are complete.

First add failing regression tests for authorization withdrawal, negated state changes, unknown outcomes, late success after cancellation, stale clarification overwrite, old ASR completion, partial-buffer resurrection, and cross-task result contamination.

Introduce task-scoped state, typed slot provenance, explicit current authorization, versioned plans, and an operation ledger that distinguishes cancellation requests from confirmed outcomes. Do not allow an old call record to restore a revoked plan. Do not convert an unknown operation to failed just because the user repeats a request.

Ground replies in the originating operation and validated result. Unify live and harness perception paths so live ASR cannot bypass confidence handling. Treat barge-in as a speech event that may carry a clarification answer, correction, cancellation, or new request.

Implement the LiveKit wrapper and actual FDB-v3 tool schemas and dependent chains. Do not merely rename the old flight-search tool. Preserve official benchmark/evaluator behavior and never hardcode benchmark examples or expected answers.

Add pinned dependencies/model/benchmark revisions and a clean-environment reproduction command. Run the official evaluation with its LLM judge enabled. Clearly separate old-kit scores, mocked tests, synthetic media tests, and actual FDB-v3 results.

Complete one real camera/manual troubleshooting extension using actual retrievable document content and citations. Keep simulated actions visibly labeled. Do not attach real consequential services until lifecycle safety tests pass.

After each phase, report changed files, tests added, results, unresolved limitations, and the next phase. Do not claim full-duplex readiness based only on text queues or old-kit scores.

Final assessment
Question	Answer
Does the project fit the theme conceptually?	Yes.
Does it contain useful asynchronous/interruption machinery?	Yes.
Does it fully satisfy the older PDF’s behavioral goals?	Not yet; important correctness gaps remain.
Does it satisfy the updated submission workflow?	No—not currently.
Are there major bugs despite passing tests?	Yes; 22 reproduced findings are detailed above.
Are real-world side effects safe to enable now?	No. Keep them mocked until lifecycle safety is fixed.
Should you restart from scratch?	No. Preserve useful components, but strengthen state ownership and add the correct benchmark integration.
The two highest-value actions are: fix authorization/cancellation correctness, and move onto the required LiveKit + FDB-v3 evaluation path. More “100/100” results on the older kit will not substitute for either.