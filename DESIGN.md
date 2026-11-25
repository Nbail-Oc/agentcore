# Design

## Architecture

City Care is a hospital front-desk prototype built on `agentcore`, a reusable
Python agent runtime. The runtime handles model calls, validated tools, hooks,
traces and sessions; the hospital layer supplies specialist agents and
policies for identity, confirmation, emergencies and uncertain writes.

The separation gives each layer a specific responsibility. `agentcore`
does not import hospital code or decide what an appointment means.
`agent_service` supplies tool implementations, patient context, prompts and
hooks. This makes the runtime reusable without moving hospital safety rules
into model prompts.

A chat turn binds the session to the verified request phone, checks for an
emergency, processes any pending confirmation, and otherwise routes the
message to a specialist. Booking and billing receive separate tool sets;
the information and human agents have no hospital tools.

## Runtime decisions

**Tool registry.** A registered tool consists of a name, description,
Pydantic input model and Python function. The registry generates Claude tool
definitions, rejects unknown tools, extra fields and invalid inputs, and
returns errors in a form the model can consume. Unexpected exceptions do not
expose their text to the model; registering the same name twice is rejected.
Validation establishes input shape, not patient authorization. Application calls use the shared
`execute_tool_call` pipeline, which applies hooks and tracing. The registry's
direct `execute` method bypasses that pipeline and is intended for tests and
trusted code.

**Bounded loop and model calls.** The loop calls `client.messages.create`
directly and returns tool results to the model until it finishes or reaches a
limit. It distinguishes normal completion from token exhaustion, refusal,
context exhaustion, API failure, timeouts and unexpected stops. Model calls
use per-attempt timeouts and bounded retries with backoff for transient
errors; SDK retries are disabled so the two retry policies do not multiply
each other. The synchronous SDK timeout is not a hard wall-clock guarantee.
The whole-run deadline is cooperative: a call already running cannot be
interrupted, and a late plain-text answer can still be returned as `done`.
The step limit counts model calls; tool calls requested on the final allowed
step are not executed. Tool results are paired with their tool-use IDs for
the supported protocol flow. A provider `pause_turn` continues the model
conversation; it is separate from a human approval pause.

**Hooks.** Before-tool hooks can allow, block or park a validated call for
confirmation; after-tool hooks can inspect or replace results. Model hooks
run before and after model calls. A crashing or invalid decision hook fails
closed. A `before_model` confirmation can pause the runtime, but application
code must manage resumption and remaining limits. `after_model` confirmation
does not provide the same resumable path, and the chat flow does not use
model-call pauses. The application must preserve and verify the approved
request and prevent replay; the runtime does not enforce those guarantees
across resumed calls.

**Tracing and state.** A trace records model attempts and tool calls,
durations, token usage and estimated cost where available. Unknown costs are
marked incomplete rather than treated as zero. A small session-store
interface isolates storage from the loop. The supplied in-memory store binds
a session to its first owner and serializes requests for that session within
one process. Binding prevents a different supplied phone from reusing that
session; it does not independently verify the phone. The store loses its
data on restart and its locks do not coordinate workers. Trace redaction is
best effort, and cost is an estimate: cached-token pricing is not calculated
separately and a known subtotal is not necessarily total provider spend.

The runtime exposes mechanisms, not hospital authority. The application
chooses which tools to expose, supplies the trusted phone context and applies
ownership and write rules. A fingerprint identifies the proposed tool and
arguments; it is not itself authentication or patient approval.

## Model decisions and code decisions

The model interprets the patient's request, chooses from its permitted tools,
supplies arguments and drafts conversational text. The application determines
which phone identifies the patient, which tools a specialist can use, whether
a proposed write is permitted, what details are read back and how a verified
write outcome is reported.

For an appointment or payment-link write, the policy checks the proposed
action and parks it with a fingerprint of the tool and arguments. The
application presents code-built details to the patient. The next patient
message must pass both a conservative code check and a model confirmation
check. The application then executes the stored call through the same tool
pipeline, rechecking the policy and relevant hospital state. A different
action requires a new confirmation.

Emergency handling precedes confirmation and routing on every turn. Strong
phrase matches and a model check contribute to the decision; an unverified
check fails conservatively. Cancellation and rescheduling use the hospital
clock for the four-hour rule. General hospital facts come from
`hospital_info.md`.

These are safeguards, not proof that the system is safe for all patients.
For example, the model can still propose the wrong valid doctor or slot;
the patient sees the proposed details, but confirmation is not a substitute
for correct interpretation. Response filters and multilingual emergency
coverage are also incomplete.

## Handling unreliable hospital writes

The hospital can return an error after saving a write. A generic retry policy
would risk duplicate or conflicting actions, so recovery depends on the
operation and on what the response establishes.

| Operation | Recovery decision |
|---|---|
| Booking | Use one idempotency key for the confirmed operation and reuse it for retries. A matching same-key successful response can establish the outcome. An ambiguous result is reported as unknown when it cannot be established. |
| Reschedule or cancel | These operations have no idempotency key. The service checks current hospital state and distinguishes a request known not to have been sent from one that may still be running. It avoids blindly retrying an in-flight request. A matching state establishes the current state, not necessarily which actor caused it. |
| Payment link | A request known not to have been sent can be retried. After an uncertain outcome, the service does not blindly retry: the API offers no link-listing operation to rule out a duplicate. |
| Slot race | Report that the original action did not happen. If a suitable alternative is available, present it as a new proposal requiring its own confirmation. |

For the supplied mock, reschedule and cancellation recovery may retry once
after a 5xx response and state checks. This relies on the mock's handling
order; a real hospital API would need an explicit completion or idempotency
contract before using the same rule. For example, a timeout can occur after a
write was saved, and a matching appointment read does not establish which
request caused the change.

An idempotency key names one intended booking across retries. The booking
handler reuses the operation's key and checks the successful response against
the requested doctor, time, patient and phone. This depends on the mock's
same-key behavior; it is not proof that a different hospital API provides
the same guarantee. For reschedule, cancel and payment link, the supplied API
does not offer the equivalent key, so their recovery paths differ.

The slot race is a separate problem: availability is a snapshot, not a
reservation. If another booking wins before this write, the service handles
the rejection and can propose a fresh alternative. The patient's original
approval does not authorize that changed time. These measures handle the
tested paths; they do not provide cross-session or cross-worker serialization.

Unresolved writes are tracked in the session to restrict conflicting
follow-up operations. An in-flight marker is saved before sending a confirmed
write. These protections depend on the store: the supplied in-memory
implementation loses them on process restart and does not coordinate multiple
workers. An unknown outcome is reported as unknown and flagged for handoff;
the flag itself does not contact staff.

The write policy checks some recognized time constraints in the latest
patient message. Constraints from earlier messages, unfamiliar wording and
other languages can be missed. A race alternative requires a new read-back
and a clear yes.

Payment-link validation checks for an HTTPS URL but does not enforce an
approved host. A real integration needs an agreed hospital payment-domain
contract and validation appropriate to that contract.

## Models, measurements and evaluation

The router, emergency and confirmation checks, and specialists use
`claude-haiku-5-5` by default. `SPECIALIST_MODEL` can select a different
specialist model.

The three recorded historical chaos runs used 130, 132 and 127 model calls,
with estimated agent costs of $0.0213, $0.0225 and $0.0210. One smoke test
measured approximately 0.8–1.0 seconds per model call. Sonnet was not run
for a measured comparison. Using only the assignment's listed token prices,
Sonnet is approximately 20 times Haiku's per-token price.

The recorded runs passed 10/13, 10/13 and 12/13 scenarios. They predate
subsequent runtime and service changes. I could not complete three new chaos
runs on the final revision because the supplied API key reached its usage
limit. Its live pass rates are therefore unknown. See `EVALS.md` for the
recorded failures and coverage limits.

## Phone calls and production gaps

For phone calls, I would add spoken read-backs, confidence-aware speech
recognition, interruption handling and a re-ask when consent is unclear.
Streaming and fewer sequential model calls would reduce perceived latency.
Caller ID would need stronger identity verification before allowing writes.

Before use with real patients, the system needs durable, shared session and
unresolved-write state; authenticated and access-controlled traces; real
staff notification and a handoff queue; monitoring and rate limits; broader
multilingual emergency evaluation; security review; and concurrency testing
under real load. The trace endpoint is open unless `TRACE_TOKEN` is set, and
its redaction is not a complete privacy guarantee.
