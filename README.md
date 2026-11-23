# City Care agent: starter repo

Starter code for the **agentcore** assignment. Read [`context.md`](context.md) first; the full brief is on the assignment page we sent you.

## What's here

| Path | What it is |
|---|---|
| `context.md` | The product, the hospital rules, the hospital system, models and cost, evals. Read first |
| `hospital_info.md` | The only facts the info agent may use |
| `mock_hospital/` | A fake hospital system (appointments and billing) with a chaos mode. Read it, don't change how it behaves |
| `agentcore/` | The reusable agent runtime (tool registry, loop, model calls with retries, hooks, trace, session store). Imports no hospital code |
| `agent_service/` | The FastAPI app (`POST /chat`, `GET /trace/{session_id}`) and the hospital agents built on `agentcore` |
| `evals/` | 10 scenarios and a runner that checks what really happened |
| `tests/` | Example tests and `fake_claude.py`, a scripted Claude client so tests run without a key |
| `tools/pack_sessions.py` | Packs your Claude Code transcripts for hand-in (see [Hand-in](#hand-in)) |

This version replaces the naive agent with `agentcore/` and a router plus booking, billing, info and human agents. See [How this version works](#how-this-version-works), [DESIGN.md](DESIGN.md) and [EVALS.md](EVALS.md).

## Your own copy

Clone or fork this repo into your own GitHub repo, and work there. To clone it into a new repo, create an empty repo on GitHub, then:

```bash
git clone <this repo's URL> citycare-agent
cd citycare-agent
git remote set-url origin <your new repo's URL>
git push -u origin main
```

Your copy can be **public**, or **private** with the **Fire-Dolphin** account invited as a collaborator (Settings → Collaborators → Add people). Either way, you send us its link when you reply to our email. Commit and push as you go.

## Setup

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env   # then put your API key in .env
```

Or with pip:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
```

(With pip, drop the `uv run` prefix from the commands below and activate the venv instead.)

Use the API key we sent you for both Claude Code and your agents. In your terminal, before starting Claude Code:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Run it

Terminal 1, the hospital system (calm):

```bash
uv run uvicorn mock_hospital.app:app --port 8001
```

The same in chaos mode (slow, flaky, races):

```bash
CHAOS=1 uv run uvicorn mock_hospital.app:app --port 8001      # PowerShell: $env:CHAOS="1"; uv run uvicorn ...
```

Terminal 2, the agent:

```bash
uv run uvicorn agent_service.app:app --port 8000
```

Try it:

```bash
curl -s localhost:8000/chat -H "content-type: application/json" -d '{"session_id":"t1","phone":"9000000001","message":"I need a skin doctor on Friday evening"}'
```

## Evals

With both services running:

```bash
uv run python -m evals.run_evals                         # every scenario, once
uv run python -m evals.run_evals --runs 3 --chaos        # what we ask you to report
uv run python -m evals.run_evals --only emergency_hinglish
```

The runner resets the hospital before each scenario and turns chaos on or off itself, so you don't need to restart the hospital between runs.

## Tests

```bash
uv run pytest
```

They must pass without an API key (about 370 tests, no network: the model is `tests/fake_claude.py`, the hospital is a scripted fake or the real mock app run in-process).

## How this version works

```
POST /chat -> emergency check -> pending confirmation? -> router (Haiku) -> booking | billing | info | human agent
                                                                              |
                                  agentcore loop: model -> tool -> hooks -> tool -> model ...
                                                                              |
                                                                       mock hospital
```

- `agentcore/`: `tools.py` (Pydantic-validated registry), `loop.py` (step limit, every stop reason, one shared tool-call path), `model.py` (per-attempt timeout, retry with backoff, SDK retries turned off), `hooks.py` (allow / block / confirm), `trace.py`, `session.py` (swappable store).
- `agent_service/`: `emergency.py`, `router.py`, `agents.py` (prompts and tool sets), `tools.py` (per-write recovery), `hooks.py` (own data, 4-hour rule, confirmation gate), `confirm.py`, `filters.py`, `replies.py`, `orchestrator.py`.
- Safety is enforced in code: tools take no phone number (the verified `/chat` phone is the only identity); every write is parked until the patient clearly confirms the exact read-back, then re-checked (ownership, state, hospital clock, 4-hour rule) and run by code; an emergency check runs first on every message.
- Writes whose result is unknown are reported as unknown (never as done or failed), never blindly retried, and flag a handoff. A handoff is only a flag in the response: nothing pages staff.

Run the agent with `uv run uvicorn agent_service.app:app --port 8000` (needs `ANTHROPIC_API_KEY`; the specialist model can be changed with `SPECIALIST_MODEL`). Inspect a conversation with `curl localhost:8000/trace/<session_id>`: every model call and tool call with inputs, outputs, time, tokens and cost (phone numbers are masked).

After an eval run, while the agent is still running, print pass/fail, failures and the measured agent cost per scenario:

```bash
uv run python tools/eval_report.py evals/results/<file>.json
```

## Claude Code sessions

We read your sessions as part of the review, so:

- **Always start Claude Code from this folder** (the repo root). Claude Code files transcripts by folder.
- To pick up where you left off the next day: `claude --continue`.
- **Before you `/clear` or close a session, export it** into `claude-sessions/`, numbered in order:

```
/export claude-sessions/01-planning.txt
/export claude-sessions/02-agentcore-loop.txt
```

- Don't edit exports or transcripts, except to remove an API key from an export.

## Hand-in

Before the deadline, run:

```bash
uv run python tools/pack_sessions.py
```

It zips the full transcripts Claude Code keeps on your machine into `claude-sessions/transcripts.zip`, including the parts of long sessions that Claude Code compacted. It also stops if it finds an API key anywhere in the repo. Then commit and push.

Your repo must contain:

| Path | What |
|---|---|
| `agentcore/` and your agents | The code |
| `README.md` | Updated so we can run your version |
| `CLAUDE.md` | Plus any other files you wrote to guide Claude Code |
| `DESIGN.md` | See the assignment page |
| `EVALS.md` | The pass rate of each of your 3 chaos runs, and why each failing case fails |
| `evals/reported/` | The 3 result files (from `evals/results/`) that `EVALS.md` reports |
| `evals/scenarios/` | The 10 given scenarios, unchanged, plus at least 3 of yours (`11_...yaml` and up) |
| `claude-sessions/` | One `/export` `.txt` per session, plus `transcripts.zip` |
