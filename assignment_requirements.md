# Assignment requirements supplement

This supplements `context.md` and `README.md`; the original assignment and email remain authoritative.

## Runtime requirements

- Register Python tools with Pydantic input models; produce Claude tool definitions, validate inputs and return tool errors to the model.
- Implement the loop using `client.messages.create`: step limit, timeouts, retry with backoff for rate limits or outages, and handling for every stop reason including token exhaustion and refusal.
- Provide hooks around model and tool calls that can allow, block with a reason, or pause for user confirmation.
- Trace each model and tool call with inputs, outputs, duration, tokens and cost; expose session traces at `GET /trace/{session_id}`.
- Put session storage behind a small interface that can be replaced without changing the loop; in-memory storage is acceptable.

## Design and delivery

- `DESIGN.md` should be 1–2 pages: runtime design and rationale; model versus code decisions; failure handling; model choices supported by cost and speed numbers; eval results and remaining failures; adaptations for phone calls; and what is missing before real patients use it.
- Keep `EVALS.md` honest: report each of the three chaos-run pass rates and explain each failing case; copy those three result files into `evals/reported/`. The rest of the hand-in checklist is in `README.md`.
- Deadline from the email: Wednesday 25 November 2026, 9:00 pm IST. Reply before then with the repository link and a 5–10 line summary covering what works, what was skipped and why, the three pass rates, and approximate hours spent. After replying or the deadline, whichever comes first, do not commit or push.
- Use Sonnet in Claude Code and check `/cost`; the $30 credit covers Claude Code, agents and evals. Use the supplied key, never include it in prompts or commits, and export each session before clearing or closing it.
