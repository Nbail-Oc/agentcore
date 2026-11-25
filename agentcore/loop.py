"""The agent loop: ask the model, run the tools it asks for, send results back, repeat."""

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .hooks import Allow, Block, Confirm, Hooks, PendingCall, ToolCall, first_decision
from .model import CallConfig, ModelCallError, call_model, response_text
from .tools import ToolRegistry, ToolResult
from .trace import Span, Trace

log = logging.getLogger(__name__)

# Anything not listed here (and not tool_use / pause_turn) becomes "unexpected_stop".
STATUS_FOR_STOP_REASON = {
    "end_turn": "done",
    "stop_sequence": "done",
    "max_tokens": "max_tokens",  # reply may be cut off: callers must not show it as a normal answer
    "refusal": "refusal",
    "model_context_window_exceeded": "context_exceeded",
}


@dataclass
class AgentResult:
    """Only treat `text` as the answer when status == "done". Do not store `messages` otherwise.

    Other statuses: blocked (a model hook; text is the reason), paused (a before_model hook asked for
    confirmation: `model_pending` describes it, no model call was made, `pending` still holds any parked tool
    call), step_limit, timeout, api_error,
    max_tokens, refusal, context_exceeded, unexpected_stop.
    """

    text: str
    messages: list[dict]  # messages added by this run, in order
    status: str
    trace: Trace
    pending: PendingCall | None = None  # a tool call parked by a Confirm hook, waiting for the user
    model_pending: PendingCall | None = None  # status "paused": the before_model request waiting for approval

    @property
    def steps(self) -> int:
        """Model calls this run used. A resumed run should get max_steps minus this."""
        return sum(m["role"] == "assistant" for m in self.messages)


def _model_block(decision) -> Block | None:
    """Model hooks can only allow or block. Confirm is for tool calls: here it fails closed, not open."""
    if isinstance(decision, Confirm):
        return Block("A safety check failed, so this was not done.")
    return decision if isinstance(decision, Block) else None


def execute_tool_call(
    registry: ToolRegistry, name: str, raw_input: Any, ctx: Any, trace: Trace, hooks: Hooks | None = None,
) -> ToolResult:
    """The one path every tool call takes, whether the model asked for it or the app resumes it.

    validate input -> before_tool hooks -> run -> after_tool hooks, with one trace span.
    """
    hooks = hooks or Hooks()
    started = time.perf_counter()
    prepared = registry.prepare(name, raw_input)
    if isinstance(prepared, ToolResult):
        result = prepared
    else:
        tool, args = prepared
        call = ToolCall(name, args)
        before, decision = call.fingerprint, Allow()
        for hook in hooks.before_tool:  # one hook at a time: a later hook must not undo an earlier one's edit
            decision = first_decision([hook], call, ctx)
            if call.fingerprint != before:  # args is the object the handler gets: hooks must not edit it
                decision = Block("A safety check failed, so this was not done.")
            if not isinstance(decision, Allow):
                break
        if isinstance(decision, Block):
            result = ToolResult(decision.reason, True)
        elif isinstance(decision, Confirm):
            parked = PendingCall(name, call.input, call.fingerprint, decision.details)
            result = ToolResult(
                "NOT EXECUTED YET. The patient must confirm first. Ask them one short question to confirm; "
                "the exact details will be shown to them automatically. Do not say it is done.", False, parked)
        else:
            result = registry.run(tool, args, ctx)
            for hook in hooks.after_tool:
                try:
                    replaced = hook(call, result, ctx)
                    result = replaced if isinstance(replaced, ToolResult) else result
                except Exception as e:  # keep the real result; a crashing hook must not hide a write's outcome
                    log.error("after_tool hook crashed: %s", type(e).__name__)
    trace.add(Span("tool", name, raw_input, result.content, time.perf_counter() - started, is_error=result.is_error))
    return result


def run_agent(
    *, client: Any, model: str, system: str, registry: ToolRegistry, messages: list[dict],
    ctx: Any = None, max_tokens: int = 1024, max_steps: int = 8, max_seconds: float | None = None,
    trace: Trace | None = None, config: CallConfig = CallConfig(), hooks: Hooks | None = None,
) -> AgentResult:
    """max_steps counts model calls. At the limit, tools the model just asked for are NOT run.

    max_seconds bounds the whole run, checked between steps: before each model call and before each tool call.
    No new work starts after the deadline, but finished work is kept: a plain answer that arrives late is still
    returned as "done", while a late tool-use answer returns "timeout" without running its tools. A call already
    running (a model request or a tool) is not interrupted, so the run can end later than max_seconds.

    Resume after status "paused": call run_agent again with `messages + result.messages` (the completed work, so
    nothing is replayed) once the app has approved. max_steps counts per call, so pass what is left. An
    after_model Confirm is not resumable (the response is already made), so it blocks instead.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    trace = trace if trace is not None else Trace()
    hooks = hooks or Hooks()
    pending: PendingCall | None = None
    deadline = config.clock() + max_seconds if max_seconds is not None else None
    tools = registry.definitions()
    new: list[dict] = []
    for step in range(max_steps):
        if deadline is not None and config.clock() >= deadline:
            return AgentResult("", new, "timeout", trace, pending)
        request = {"model": model, "system": system, "messages": messages + new,
                   "tools": [d["name"] for d in tools], "max_tokens": max_tokens}
        decision = first_decision(hooks.before_model, copy.deepcopy(request), ctx)  # a copy: hooks decide, they do not edit
        if isinstance(decision, Confirm):  # nothing was sent yet: park the request; resume = run_agent again once approved
            digest = hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()
            parked = PendingCall("model", {"model": model, "tools": request["tools"]}, digest, decision.details)
            return AgentResult("", new, "paused", trace, pending, parked)
        blocked = _model_block(decision)
        if blocked:
            return AgentResult(blocked.reason, new, "blocked", trace, pending)
        try:
            response = call_model(
                client, trace, model=model, system=system, tools=tools, messages=messages + new,
                max_tokens=max_tokens, config=config, deadline=deadline,
            )
        except ModelCallError as e:
            return AgentResult("", new, "timeout" if e.reason == "DeadlineExceeded" else "api_error", trace, pending)
        blocked = _model_block(first_decision(hooks.after_model, copy.deepcopy(response), ctx))
        if blocked:
            return AgentResult(blocked.reason, new, "blocked", trace, pending)
        reason = response.stop_reason
        if reason == "tool_use" and not any(b.type == "tool_use" for b in response.content):
            return AgentResult(response_text(response), new, "unexpected_stop", trace, pending)  # nothing to answer
        new.append({"role": "assistant", "content": response.content})
        if reason == "tool_use":
            if step == max_steps - 1:
                break
            if deadline is not None and config.clock() >= deadline:  # answer came too late: start nothing
                new.pop()
                return AgentResult("", new, "timeout", trace, pending)
            results, parked_now = [], False
            for block in response.content:
                if block.type == "tool_use":
                    if deadline is not None and config.clock() >= deadline:
                        out = ToolResult("Not executed: the time limit was reached.", True)
                    elif parked_now:  # one action waits for confirmation at a time: skip the rest of this response
                        out = ToolResult("Not executed: another action is waiting for the patient's confirmation.", True)
                    else:
                        out = execute_tool_call(registry, block.name, block.input, ctx, trace, hooks)
                        if out.pending is not None and pending is not None:
                            out = ToolResult("Not executed: only one action can wait for confirmation at a time.", True)
                        elif out.pending is not None:
                            pending = out.pending
                            parked_now = True
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": out.content, "is_error": out.is_error,
                    })
            new.append({"role": "user", "content": results})
        elif reason != "pause_turn":  # pause_turn: send the turn back unchanged so the model can continue
            status = STATUS_FOR_STOP_REASON.get(reason, "unexpected_stop")
            return AgentResult(response_text(response), new, status, trace, pending)
    return AgentResult(response_text(response), new, "step_limit", trace, pending)
