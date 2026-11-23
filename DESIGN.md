# DESIGN

## agentcore and why it is shaped this way

`agentcore/` is a small runtime with no hospital code (a test checks its imports). A **tool registry** turns Python functions with Pydantic inputs into Claude tools, rejects unknown fields and bad input before the tool runs, and turns every failure into a result the model can read. The **loop** (`client.messages.create`, no framework, no tool runner) has a step limit, a whole-run time limit, a status for every stop reason (a truncated or refused reply is never treated as an answer) and one shared `execute_tool_call` path. **`call_model`** gives each attempt a timeout, retries rate limits, 5xx and connection errors with capped backoff and turns the SDK's own retries off so they cannot multiply ours. **Hooks** run before and after every model and tool call and can allow, block with a reason, or *confirm*, which parks the call so the app can run it later through the same path. A hook that crashes blocks the call. **Traces** record every attempt with inputs, outputs, time, tokens and cost, and spans are scrubbed (API keys, a redact hook for phone numbers). The **session store** is an interface (in-memory now) with per-session locks and first-owner-wins binding.

## What the model decides, what code decides

The model understands the patient, picks tools and arguments, and writes friendly text. Code decides everything that can hurt someone: who the patient is (the `/chat` phone only; no tool accepts a phone), whether a write may run (own data, current state, hospital clock, 4-hour rule), whether the patient clearly confirmed the exact action, what the read-back says (built from the tool arguments), what is reported after a write (built from the verified result), and whether a message is an emergency (code phrases first). Confirmation needs two independent checks: a whole-word code check (explicit yes, no negation, question or extra request; English, Hindi, Hinglish) and a small model verdict. A pending action lives for exactly one patient message.

## Failure handling

- **Hospital errors**: classified as definite (4xx), not sent, or unknown (timeout after send, any 5xx: a 503 can follow a saved write). Reads retry; writes never retry in the client.
- **Booking**: one idempotency key per confirmed operation (new for every new operation), same-key retries, then a lookup for evidence.
- **Reschedule and cancel** (no key): check the hospital's state, recheck once, retry once. A definite error after an unknown try is not believed on its own, because the first request may still land.
- **Payment link**: never retried after an unknown result (duplicates cannot be ruled out and there is no way to list links).
- Finding nothing is never proof of failure: the outcome stays *unknown*, the chat is flagged, and the reply says it could not be confirmed. Only 2xx or the target state counts as success; only a definite error counts as failure.
- **Model outage**: two failed turns in a row flag a handoff. **Emergency check failure**: conservative reply, no writes.
- A **handoff is only a flag** in the response. Nothing pages staff, so replies say the chat is flagged and give the front desk number; they promise no contact.
- Two filters catch obvious mistakes (numbers in info replies must come from `hospital_info.md`; no dosages or "it's probably nothing"). They are nets, not guarantees.

## Models, cost and speed

Router, emergency, confirmation checks and all four agents use `claude-haiku-5-5` with thinking turned off. Measured from traces: a calm run of 13 scenarios was 125 model calls for $0.0195; each of three chaos runs was about 130 calls for about $0.021; a model call took about 0.8 to 1.0 s. I did not run Sonnet, so there is no measured comparison; by price alone it would be about 20 times more per token. I kept Haiku because the failures in `EVALS.md` are not ones I expect a bigger model to fix by itself (they are about confirmation flow and wrong tool arguments), but that is an untested guess.

## Evals

Three chaos runs: 10/13, 10/13, 12/13 (see `EVALS.md`). Scenario 03 fails in all three: after a lost race the next slot is offered with a read-back, but the patient's "yes, book that one" is not accepted; cause not yet confirmed. Others: a wrong doctor id in a tool call, a lost race with no turn left to confirm, an unknown payment-link result (by design not retried), a name the model re-asked for, and a phrase-match artifact in scenario 09.

## Phone calls instead of chat

Confirmation would need spoken read-backs and a spoken yes, so the whole-word text check would be replaced by speech transcripts with confidence and a re-ask on low confidence. Latency matters far more (about 3 model calls per turn is too slow: merge the emergency and routing checks, stream replies, cache the prompt). The emergency check must work on partial transcripts and interrupts. Caller identity from caller ID is weaker than a verified WhatsApp number, so writes would need a second factor.

## Missing before real patients

Real staff paging and a handoff queue (today only a flag); authentication on `/trace` (anyone with a session id can read it; phones are masked but messages are not); a durable session store with expiry; rate limiting and abuse controls; Marathi and wider emergency-phrase coverage with measured recall; a real evaluation set with many runs per scenario; monitoring and alerting on unknown write outcomes; concurrency tests with real load (writes from two chats of one patient can interleave: the hospital, not this code, arbitrates); a security review of prompt-injection paths through stored patient names and tool text; and review of the cost caps.
