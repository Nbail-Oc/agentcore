# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## CityCare project

Build a reusable, hospital-independent Python agent runtime and specialist agents for City Care Hospital's fictional WhatsApp front desk. Preserve the `/chat` contract. Enforce safety in code, handle uncertain hospital writes, and verify behavior rather than relying on prompts or claims.

- Read `context.md` and `assignment_requirements.md` before implementation. The assignment's explicit requirements take precedence over generic simplicity guidance.
- Do not modify `mock_hospital/`, the 10 supplied scenarios, or existing eval-check behavior. Do not use agent frameworks or the SDK tool runner.
- Use the verified request phone for identity; require a specific read-back and clear confirmation for writes. Check emergencies before routing and use the hospital clock for the four-hour rule.
- Keep `agentcore/` independent of hospital code. Test safety and failure paths with the fake client; keep tests runnable without an API key.
- Treat each write's retry and recovery semantics separately. Confirm outcomes from available hospital state; do not infer success from a failed or ambiguous response.
- Bind session state and tool access to the verified phone; consider concurrent requests and prevent cross-patient data leakage.
- Distinguish code findings from hypotheses and measured results. Do not claim an eval passed or a failure occurs without evidence; preserve time for the mandatory router, specialists, three chaos runs and hand-in.
- Never expose or commit secrets. Work in small, verified increments and explain material trade-offs and failures.

## Context and references

- `context.md`: product, API contract, hospital behavior, safety rules, models and evals; read before design or coding.
- `assignment_requirements.md`: runtime requirements and design/hand-in details absent from the starter context; consult when planning and before submission.
- `hospital_info.md`: sole source for general hospital facts; consult for info-agent answers.
- `README.md`: setup, commands, tests, session exports and submission checklist; consult when running or handing in.
- `../CONTEXT/Agent_Core_Assignment.md` and `../CONTEXT/mail.md`: original full assignment and deadline email, if available; consult for authoritative details.
- `agent_service/schemas.py`: `/chat` request and response fields; consult when changing the service.
- `mock_hospital/`, `evals/`, `tests/fake_claude.py`: hospital behavior, evaluation cases and offline model test double; read when implementing or testing, subject to protected-file rules above.
