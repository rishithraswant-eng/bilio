> **ARCHIVAL** — superseded by docs/SUBMISSION_CHECKLIST.md; paths below may not exist.

# Guideline alignment & audit tracker

This file tracks two things:

1. How BILIO meets the **updated Theme 05 participant guide**
   (`docs/guides/Theme05_Participant_Guide_UPDATED.txt`). That guide replaces the older queue-based kit
   (`docs/guides/Theme5_Guide_OLDER_KIT.txt`).
2. The status of every finding in the independent audit (`docs/audit/AUDIT_R01-R22.md`).

The updated guide is the submission target. The older kit, meaning `harness/`, `scenarios/` and `run_local.py`,
is kept unchanged. We still use it as a regression suite. Its scores are labelled **old-kit** everywhere and
are never reported as FDB-v3 results.

## 1. Updated guide → where it is satisfied

| Guide requirement | Where | Status |
|---|---|---|
| LiveKit voice agent (custom architecture allowed) | `fdb_v3/bilio_agent.py` | see §3 |
| Cascaded or realtime pipeline following the benchmark's templates | `fdb_v3/bilio_agent.py` (Silero VAD + STT + BILIO coordinator + TTS) | see §3 |
| FDB-v3 12 tools across 4 domains, logged in the official telemetry format | `fdb_v3/tools.py` | see §3 |
| One-command reproduction (install, configure, evaluate) | `scripts/reproduce_fdb_v3.sh` | see §3 |
| Declared model provider | `submission.yaml`, `README.md` | see §3 |
| Pinned seeds and versions | `requirements-fdb.txt`, `fdb_v3/PINNED.md`, `submission.yaml` | see §3 |
| Results and run logs from our best run | `results/` | see §3 |
| One extension beyond the benchmark domains, working end to end | `extension/` (camera + manual troubleshooting with page citations) | see §3 |
| README: architecture diagram, setup, extension clearly marked | `README.md` | see §3 |
| Slide deck, at most 8 slides | `submission/slides/` | see §3 |
| Demo video, 3 to 5 minutes | `submission/VIDEO_SCRIPT.md`; the recording itself is made by the team | see §3 |
| No hard-coding or memorising of benchmark items | enforced by `tests/test_no_benchmark_leakage.py` | see §3 |
| No calls to our own servers at eval time | agent logic lives in-repo; only the declared hosted APIs are used | see §3 |
| No caching across scenarios | a fresh coordinator is built per LiveKit room/job | see §3 |

## 2. Audit findings R01–R22

Status legend: **fixed** means fixed with a regression test in `tests/test_audit_r.py`. **partial** means partly
addressed; see the note. **open** means not yet addressed.

| ID | Pri | Finding | Status |
|---|---|---|---|
| R01 | P0 | Revoking booking permission does not stop the plan | open |
| R02 | P0 | Negated non-booking actions are executed | open |
| R03 | P0 | Cancellation treated as proof a side effect did not happen | open |
| R04 | P0 | Unknown outcome becomes retryable on repetition | open |
| R05 | P1 | Rebooking after confirmed cancellation returns stale "already booked" | open |
| R06 | P1 | Clarification answers overwritten by the original utterance | open |
| R07 | P1 | Partial text buffered before a correction restores stale intent | open |
| R08 | P1 | Old ASR becomes a new request after a newer turn | open |
| R09 | P1 | Corrections to origin / arbitrary schema fields ignored | open |
| R10 | P1 | Lowercase passenger names silently reuse an earlier passenger | open |
| R11 | P1 | New booking request mistaken for a short search follow-up | open |
| R12 | P1 | Flight-selection constraints ignored (cheapest / ordinal) | open |
| R13 | P1 | Known + unknown city extraction don't combine | open |
| R14 | P1 | Enum self-correction picks schema order | open |
| R15 | P1 | Argument construction doesn't enforce schema constraints | open |
| R16 | P1 | Malformed "success" results produce false completion claims | open |
| R17 | P1 | Newer camera frame still yields lookup grounded in old frame | open |
| R18 | P1 | Live server ASR bypasses uncertainty handling | open |
| R19 | P1 | Clarification answered while the assistant speaks is lost | open |
| R20 | P1 | Tool-provider exceptions strand live operations | open |
| R21 | P2 | Filler deduplication causes silence on later turns | open |
| R22 | P1 | Concurrent tasks contaminate each other's completion messages | open |

### Operational findings (audit §6)

| Finding | Status |
|---|---|
| Server learns about barge-in too late | open |
| Queued server speech has no task/version tag | open |
| SSE can lose events (no sequence / replay) | open |
| Session cleanup incomplete (reaper, media retention, graceful stop) | open |
| Unbounded live ASR workload | open |
| Malformed HTTP input (non-object body, negative Content-Length) | open |
| Recorder timeout may stop a later recording | open |
| Model initialisation not locked, failures sticky, revisions unpinned | open |
| README states media absent / unclear provenance / missing eval_submission.py | open |

## 3. Progress log

Each phase ends with a commit, and the PR is updated after it.
