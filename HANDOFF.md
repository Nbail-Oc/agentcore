# HANDOFF for a new Claude Code session (internal notes, not part of the submission)

Written 2026-11-24 from a live look at the repo (git, files, offline tests). Labels: **VERIFIED** (checked in this session just now), **ESTIMATE** (computed, not exact), **UNRESOLVED** (needs a decision or a check). Start Claude Code from the repo root `C:\Users\thang\Downloads\agentcore\citycare-agent`. Do not edit `CLAUDE.md`, `mock_hospital/`, the 10 scenarios `evals/scenarios/01..10_*.yaml` or `evals/run_evals.py`. Delete or `.gitignore` this file before the final commit.

## 1. Git state (VERIFIED)
- Branch `main`, HEAD **`f1034dc`** "fix(agent_service): propose the nearest free time in code after a lost race". Its parent is **`bb127e7`** "Updated Sys Prompts" (author NbailOc, amended). This history was rewritten after my last session: the reflog shows checkout `73e0adc`, `commit --amend` (-> `bb127e7`), then a cherry-pick of the lost-race fix.
- The amend only removed 6 session export files from the old prompts commit `3abd3ea`. The four prompt files (`agent_service/agents.py`, `router.py`, `emergency.py`, `confirm.py`) are byte-identical to `3abd3ea`.
- **`origin/main` is still `3abd3ea`. Local `main` and `origin/main` have diverged** (1 commit only on origin, 2 only local). A normal push is rejected; making them match needs a force-push or a reset. The assignment forbids force-pushes **after** the reply email. UNRESOLVED: which way to reconcile, and when.
- **My four commits from the last session are NOT on any branch** but still exist as objects (reflog `HEAD@{7}`..`HEAD@{4}`): `a9071e2` nudge guard + info wording, `7d4fd99` session-limited alternatives + doctor ids in the booking prompt, `becb63a` booking verification snapshot + last-write memory + booking recheck, `cb8ea6e` optional bill/appointment ids resolved in the hook. Their parent is `e1a40cd` (the original of `f1034dc`). Recover with `git branch recovered-steps-A-D cb8ea6e` (not done), then `git cherry-pick a9071e2 7d4fd99 becb63a cb8ea6e` onto `main` if wanted. UNRESOLVED: was dropping them intentional?
- The prompt text in `bb127e7` was written by user manually. Provenance is UNRESOLVED with the team. `agent_service/confirm.py` line 88 has an example that mirrors scenario 03's last turn; `agent_service/emergency.py` has a stray backslash in the prompt.
- Uncommitted (VERIFIED): `.claude/settings.json` and `skills-lock.json` modified; untracked `.agents/skills/handoff/`, `.claude/skills/handoff/`, `claude-sessions/01-session-1.txt`, `evals/reported_new/`, `evals/reported_next/`; tracked `claude-sessions/chaos-eval-done.txt` is deleted in the working tree; many old `.claude/skills/*` files show as deleted. `claude-sessions/` now holds only `01-session-1.txt` (the earlier numbered exports are gone from disk; why is not verified).
- Offline servers: the agent on port 8000 was down when checked; the mock on 8001 not confirmed.

## 2. What is in HEAD (VERIFIED by file presence and the suite)
`agentcore/` (tools registry, loop with step limit and stop reasons, `model.py` timeouts/retries, hooks incl. parked confirmations, trace, session store), `agent_service/` (hospital client, per-write recovery in `tools.py`, safety hooks `hooks.py`, two-signal confirmation `confirm.py`, emergency check, router, agents, filters, handoff wording, nearest-free-time recovery in `recovery.py`), scenarios `evals/scenarios/11_..13_*.yaml`, `tools/eval_report.py`, `tests/` including `test_against_mock.py` and `test_scenario03_flow.py`.
**Offline suite at HEAD: 384 passed, `ANTHROPIC_API_KEY` unset (run just now).** With steps A-D it was 438 (not in HEAD).

## 3. Chaos results (files VERIFIED to exist; code mapping partly unverified)
- `evals/reported_next/run-20261124-030612-1.json`: **12/13 (92%)**, one run only, on `cb8ea6e` (so on code that is **not in HEAD**). Failure: `billing_then_booking`: the model asked for a full name already given; the scripted "Yes, book it." only parked the booking. Agent cost $0.0259 measured over 114 model calls. The nudge guard made 10 extra model calls in that run (5 in that scenario).
- `evals/reported_new/run-20261124-020417-1.json` 11/13, `...020932-1.json` 10/13, `...021423-1.json` 9/13: three runs on `e1a40cd`. HEAD's `f1034dc` is a cherry-pick of `e1a40cd`, so `agent_service/` should match; **not diffed**: run `git diff e1a40cd HEAD --stat -- agent_service agentcore tests evals tools`.
- `evals/reported/run-20261124-005325-1.json`, `...005747-2.json`, `...010146-3.json`: committed older set, 10/13, 10/13, 12/13, on older code (`909b29b` era).
- Failure causes seen: scenario 09 is a wording false positive of the unchanged check (hedged replies echo "Star Health"); scenario 03 failures were the model asking in prose instead of parking, or a pre-commit 503 plus a race left the outcome `unknown` (by design); wrong doctor id and "nearest free time" proposing 12:00 for an evening wish (HEAD has no session restriction; fixed only in `7d4fd99`).
- No complete set of three runs exists for any single current code version that matches HEAD or a reachable commit.

## 4. Open concerns (UNRESOLVED, none fixed in HEAD)
1. **Stale write result.** In `becb63a`, `session["last_write"]` is put in the system prompt as "authoritative". It is a snapshot of the last write's result and can go stale (a later phone cancellation, another chat, an unknown outcome that resolves). Plan: reword to "result at the time of that write; check my_appointments / my_bills for the current state" and add a test.
2. **Confirmation nudge precision.** In `a9071e2` the guard (`replies.asks_for_confirmation`, regexes `_ASKS` and `_ACTION` in `agent_service/replies.py`) re-runs the agent when a reply asks for a yes but nothing is parked. It fired 10 times in the 12/13 run. Not checked: whether it matched harmless offers ("Would you like me to send a payment link?") and could push the model to propose an unrequested write (the patient would still have to confirm). Inspect via `GET /trace/<session_id>` while the agent that ran it is alive (traces are in memory only), or add offline cases for offers.
3. A "yes" after a prose-only question books nothing (tested offline); the real fix is getting the model to call the tool (nudge) or accepting those scenario failures.

## 5. Documentation that is outdated
`EVALS.md` (reports the older 10/13, 10/13, 12/13 set and says scenario 03's cause is unconfirmed; the cause is now identified from traces), `DESIGN.md` (evals paragraph, recovery section, model-choice numbers), `README.md` ("about 370 tests"; mentions no nudge, session limits or last-write memory). `evals/reported/` holds the older set. All must match the version finally tested.

## 6. Budget (ESTIMATES; from local transcript token counts with assumed cache pricing, not account data)
- The user's cap since approving the last plan was $4 total. Estimated spent: about **$3.8**. Remaining about **$0.2**: not enough to proceed. A call made after more than 5 idle minutes re-reads the whole conversation and cost about $1.7 once; keep calls close together and start a fresh session (much cheaper context).
- Options put to the user, awaiting a reply: **A** finish on one code version with about $1.2 more (two more chaos runs, docs); **B** about $2 more (check the nudge and stale-wording concerns first, then re-run all three chaos runs). Agent-service cost per chaos run is small: $0.026 to $0.032 measured.
- Not yet approved: any further paid run. Offline tests are free.

## 7. Remaining hand-in steps
1. Decide steps A-D (recover and cherry-pick, or drop) and the prompt provenance; then settle the final code version.
2. Run three full chaos runs on that exact version (`uv run uvicorn mock_hospital.app:app --port 8001`, `uv run uvicorn agent_service.app:app --port 8000`, one `uv run python -m evals.run_evals --runs 1 --chaos` per call, ~4.5 min each); copy files to `evals/reported/` (replace the old three); `uv run python tools/eval_report.py <file>` for cost.
3. Update `EVALS.md`, `DESIGN.md`, `README.md`; commit with descriptive messages.
4. Export the session(s); run `uv run python tools/pack_sessions.py` last; commit `claude-sessions/` (including `transcripts.zip`).
5. Secret scan (`git grep -n "sk-ant-"`, expected hits only in `.env.example`, README, `tools/pack_sessions.py`, tests, `agentcore/trace.py`); fix the branch divergence; push to `origin` (repo `github.com/Nbail-Oc/agentcore.git`; public, or Fire-Dolphin invited).
6. Email before **Wed 25 Nov 2026, 9:00 pm IST** with the repo link and a 5-10 line summary (pass rate of each chaos run, hours spent). **No commits or pushes after the reply.** The email is the user's to send.
