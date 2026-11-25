# City Care agent

City Care is a hospital front-desk prototype built on `agentcore`, a reusable
Python agent runtime. The runtime handles model calls, validated tools, hooks,
traces and sessions; the hospital layer supplies specialist agents and
policies for identity, confirmation, emergencies and uncertain writes.

The service simulates a WhatsApp front desk for City Care Hospital in Pune.
Patients can ask about appointments, bills and hospital information. A
separate mock hospital API supplies the data and can introduce latency,
errors and booking races.

This is an assignment prototype, not a system ready for real patients.
See [DESIGN.md](DESIGN.md) for safety decisions and limitations, and
[EVALS.md](EVALS.md) for recorded results and their version limitation.

## Request flow

```text
POST /chat
    |
    v
Bind session to the verified request phone
    |
    v
Emergency check on every message
    |
    +-- emergency or check unverified --> safety response and handoff flag
    |
    v
Pending action + clear confirmation?
    |
    +-- yes --> recheck policy --> execute stored tool call --> report outcome
    |
    v
Haiku router
    |
    +-- booking --> doctors, slots and appointments
    +-- billing --> bills and payment links
    +-- info    --> hospital_info.md
    +-- human   --> handoff response
```

The router and specialists share the conversation history. The booking and
billing specialists receive only their respective tools; the info and human
agents receive no hospital tools.

A handoff is a flag in the response. This prototype does not notify or page
hospital staff.

If the emergency check cannot be verified, the service responds
conservatively and sets the handoff flag. It does not always set
`emergency: true` in that case.

## What `agentcore` provides

`agentcore/` is independent of hospital code. The application supplies tools,
context and policies; the runtime executes them.

| Component | Responsibility |
|---|---|
| Tool registry | Registers Python functions with Pydantic input models, generates Claude tool definitions, validates inputs and returns tool errors to the model. |
| Agent loop | Calls `client.messages.create`, executes requested tools, returns results to the model and stops on completion, limits or an explicit failure status. |
| Model calls | Applies attempt and overall time limits, bounded retries with backoff, and records each attempt. |
| Hooks | Provide checkpoints around model and tool calls. Policies can allow or block a call; a tool call can be parked for confirmation. |
| Tracing | Records model attempts and tool calls with inputs, outputs, timing, token counts and estimated cost where available. |
| Session interface | Keeps session storage behind a small interface. The included implementation is in memory. |

The loop and hospital policies are separate: a different application can
register different tools and hooks without importing City Care code into the
runtime. The included offline tests exercise this separation.

For a tool call, the runtime validates the input before running policy hooks.
A hook can park the exact proposed call; the application owns the decision to
approve and resume it. The loop limits model steps and checks its deadline
before starting more work. These checks do not interrupt a tool already
running. The in-memory session store and its locks work within one process
and do not survive a restart.

## Hospital tools

| Agent | Tools |
|---|---|
| Booking | Find doctors, check free slots, list the patient's appointments, book, reschedule and cancel. |
| Billing | List the patient's bills and request a payment link. |
| Info | No hospital tools; general facts come from `hospital_info.md`. |
| Human | No hospital tools; produces a handoff response. |

Hospital tools use the verified phone in the request context. The model
cannot supply a different phone as a tool argument.

For a write, the policy builds a read-back of the proposed action and parks
the call. A subsequent clear confirmation allows the stored call to pass
through the tool pipeline again, where ownership, current state and
applicable rules are checked. Changed details require a new read-back and
confirmation.

If a slot is taken between the availability check and the booking, the
service can propose a new slot; it requires a new read-back and confirmation.
Booking retries reuse an idempotency key so the supplied mock can recognize
the same operation. For a write whose outcome cannot be established, the
service reports uncertainty and restricts conflicting follow-up actions rather
than blindly repeating it. These protections rely on the hospital API's
contract and the current in-memory session; they are not a distributed or
restart-safe guarantee. See [DESIGN.md](DESIGN.md) for operation-specific
recovery and its limits.

## Project layout

```text
agentcore/          Reusable agent runtime
agent_service/      FastAPI service, specialists and hospital policies
mock_hospital/      Supplied hospital API and chaos behavior
evals/scenarios/    Ten supplied scenarios and three additional scenarios
evals/reported/     Three historical chaos-run result files
tests/              Offline tests, scripted Claude client and hospital fakes
tools/              Evaluation report and session-packing utilities
hospital_info.md    Source for general hospital information
DESIGN.md           Architecture, trade-offs and production gaps
EVALS.md            Results, failures and coverage limitations
```

## Setup

Requires Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env
```

Add the supplied `ANTHROPIC_API_KEY` to `.env` for the agent service.
Do not commit `.env` or an API key.

Alternatively:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python`.

## Run

Start the mock hospital in one terminal:

```bash
uv run uvicorn mock_hospital.app:app --port 8001
```

Start the agent service in another:

```bash
uv run uvicorn agent_service.app:app --port 8000
```

Try a conversation:

```bash
curl -s http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d '{"session_id":"t1","phone":"9000000001","message":"I need a skin doctor on Friday evening"}'
```

Available endpoints:

- `POST /chat` — process one conversation turn.
- `GET /health` — service health.
- `GET /trace/{session_id}` — inspect stored traces for a session.

Traces can contain patient messages and model inputs and outputs. Set
`TRACE_TOKEN` to require a bearer token for the trace endpoint. Phone
masking and key scrubbing are best effort, not a complete privacy guarantee.
When `TRACE_TOKEN` is unset, the endpoint is open to callers who know the
session ID. With a token configured, supply it privately as a bearer token:

```bash
curl -H "Authorization: Bearer $TRACE_TOKEN" "http://localhost:8000/trace/$SESSION_ID"
```

Do not put the token itself in the repository or an example command.

## Tests and evaluations

Offline tests use a scripted Claude client and do not require an API key:

```bash
uv run pytest
```

On 25 November 2026, the code at commit `75840f9` passed 592 offline tests
with the API key unset. The tests use scripted model responses and local mock
behavior; they do not establish a live chaos pass rate.

With both services running, execute the evaluation scenarios:

```bash
uv run python -m evals.run_evals
uv run python -m evals.run_evals --runs 3 --chaos
uv run python -m evals.run_evals --only emergency_hinglish
```

The evaluation runner resets the mock hospital between scenarios and
configures chaos for the run. `EVALS.md` describes the saved historical
results; those runs predate later code changes and do not establish the
final revision's pass rate.

## How I worked with Claude Code

I started by asking Claude Code to read the assignment context and inspect
the existing repository before making changes. The initial request called
for an architecture map, protected-file constraints, evidence-backed risks,
a prioritized implementation roadmap and a free-test baseline. I asked it
to distinguish verified findings from assumptions and avoid paid
evaluations without approval.

I then directed the work in narrow, verifiable increments, using offline
tests and scripted failures to examine the runtime and hospital safety paths.
`CLAUDE.md` records project-specific boundaries, including the verified-phone
identity rule, the protected mock and scenarios, and the requirement to treat
uncertain writes honestly.

The recorded sessions show CodeGraph MCP used to explore code paths and
Context7 MCP consulted for SDK documentation. They also show invocations of
the `ponytail`, `caveman` and test-driven-development skills while working.
These tools assisted exploration and development; the application does not
depend on MCP servers or those skills at runtime.

See `claude-sessions/` for the exported sessions.


## Session archive

The assignment requires Claude Code session exports and a
`claude-sessions/transcripts.zip` archive, generated from the repository root
with `uv run python tools/pack_sessions.py`. Do not edit exports or transcripts
except for the assignment's explicit API-key removal exception.
