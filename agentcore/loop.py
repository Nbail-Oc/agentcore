"""The agent loop: ask the model, run the tools it asks for, send results back, repeat."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from .hooks import Block, Confirm, Hooks, PendingCall, ToolCall, first_decision
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

    Other statuses: blocked (a model hook; text is the reason), step_limit, timeout, api_error,
    max_tokens, refusal, context_exceeded, unexpected_stop.
    """

    text: str
    messages: list[dict]  # messages added by this run, in order
    status: str
    trace: Trace
    pending: PendingCall | None = None  # a call parked by a Confirm hook, waiting for the user


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
        decision = first_decision(hooks.before_tool, call, ctx)
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
                    result = hook(call, result, ctx) or result
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

    max_seconds bounds the whole run (model calls, waits and tools, checked between steps).
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
            return AgentResult("", new, "timeout", trace)
        request = {"model": model, "system": system, "messages": messages + new,
                   "tools": [d["name"] for d in tools], "max_tokens": max_tokens}
        blocked = first_decision(hooks.before_model, request, ctx)
        if isinstance(blocked, Block):
            return AgentResult(blocked.reason, new, "blocked", trace, pending)
        try:
            response = call_model(
                client, trace, model=model, system=system, tools=tools, messages=messages + new,
                max_tokens=max_tokens, config=config, deadline=deadline,
            )
        except ModelCallError as e:
            return AgentResult("", new, "timeout" if e.reason == "DeadlineExceeded" else "api_error", trace, pending)
        blocked = first_decision(hooks.after_model, response, ctx)
        if isinstance(blocked, Block):
            return AgentResult(blocked.reason, new, "blocked", trace, pending)
        new.append({"role": "assistant", "content": response.content})
        reason = response.stop_reason
        if reason == "tool_use":
            if step == max_steps - 1:
                break
            results, parked_now = [], False
            for block in response.content:
                if block.type == "tool_use":
                    if parked_now:  # one action waits for confirmation at a time: skip the rest of this response
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
