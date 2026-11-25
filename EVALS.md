# Evaluations

## Recorded chaos runs

These are the three saved result files in `evals/reported/`. Each covers the
ten supplied scenarios and three additional scenarios. The recorded runs
used `claude-haiku-5-5` for routing, checks and specialists.

| Run | Result file | Passed | Rate |
|---|---|---:|---:|
| 1 | `run-20261124-005325-1.json` | 10/13 | 77% |
| 2 | `run-20261124-005747-2.json` | 10/13 | 77% |
| 3 | `run-20261124-010146-3.json` | 12/13 | 92% |

**Version limitation:** These runs were recorded before later changes to the
runtime and agent service. They document the version tested at the time, not
the final revision. I could not complete three new chaos runs on the final
revision because the supplied API key reached its usage limit. The final
revision's live pass rates are therefore unknown. Individual runs of a
nondeterministic system are not a stable estimate of future performance.
The exact source revision used for these historical files has not been
independently established.

## Current-code offline tests

On 25 November 2026, the code at commit `75840f9` passed 592 offline tests
with the API key unset. The test run used a scripted Claude client and local
mock behavior. It does not establish live Claude behavior or a chaos pass rate.
Documentation changes after that commit do not change the code tested.

The offline suite includes focused coverage of these design decisions:

| Area | Examples of coverage | What it does not establish |
|---|---|---|
| Runtime | Tool validation and duplicate registration; hooks and parked calls; message/result ordering; model retries, deadlines and step limits. | Live provider behavior or a hard deadline for a running synchronous tool. |
| Sessions and trace | In-process session binding and locks; isolation; estimated and incomplete cost; key-scrubbing examples. | Restart durability, multi-worker coordination, complete privacy or actual provider billing. |
| Booking and races | Same-key booking retries, a lost slot, and a newly confirmed alternative. | That a real hospital honors the mock's idempotency and race-handling contract. |
| Uncertain writes | Delayed completion, unknown outcomes, write markers and operation-specific retry decisions. | Resolution of every real API failure or recovery after losing in-memory state. |
| Patient safety | Confirmation, ownership checks, emergency priority and selected response filters. | Coverage of every wording, language or clinical emergency. |

Relevant tests include `tests/test_agentcore_loop.py`,
`tests/test_agentcore_hooks.py`, `tests/test_agentcore_session.py`,
`tests/test_hospital_tools.py`, `tests/test_delayed_completion.py` and
`tests/test_scenario03_flow.py`. These are offline tests; they are not
substitutes for the live evaluations below.

## Failures in the recorded runs

| Run | Scenario | Recorded failure and explanation |
|---|---|---|
| 1, 2, 3 | `slot_taken_while_booking` | No matching appointment was booked after the original slot was lost. The saved conversation shows another read-back instead of completing the booking. The precise confirmation failure was not established from traces for these runs. |
| 1 | `info_not_in_docs` | The reply said staff could confirm whether a named insurer was accepted. The scenario's phrase check matched the wording, although the reply did not assert that the insurer was accepted. |
| 1 | `billing_then_booking` | No payment link and no matching booking were recorded. The saved conversation indicates an uncertain payment-link result after a 503 and that the booking agent asked again for a name already given. |
| 2 | `book_new_patient` | No matching booking was recorded. The model proposed a different doctor's ID from the doctor it had discussed; the read-back named that doctor, whose relevant slots were full. |
| 2 | `reschedule_own` | The appointment remained on its original date after a slot race. The alternative required a new confirmation, but the scripted conversation had no turn left to provide it. |

The confirmation, four-hour-rule, other-patient-data and emergency scenarios
passed in all three recorded runs. The recorded saved-before-503 booking and
reschedule scenarios also passed in all three. These observations apply to
those runs; they do not establish behavior for other inputs or the final code.

## Cost and coverage

Previously reported trace totals for the historical runs were 130, 132 and
127 model calls, with estimated agent costs of $0.0213, $0.0225 and $0.0210
respectively, using the assignment's prices. These are not measurements of
the current revision and exclude Claude Code spending. Sonnet was not measured.

The recorded scenarios do not establish performance across other phrasings,
Marathi, concurrent patients or a broad emergency test set. The model retry
path was not exercised by a failed model call in these live runs. Offline
tests exercise additional paths without an API key.

When API access is available, run three live chaos evaluations against the
final code, save each result and investigate each failure. The required
current-version results remain unavailable until then.
