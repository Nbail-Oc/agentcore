"""The agent loop: ask the model, run the tools it asks for, send results back, repeat."""

import time
from dataclasses import dataclass
from typing import Any

from .tools import ToolRegistry, ToolResult
from .trace import Span, Trace, clip

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
    """Only treat `text` as the answer when status == "done". Do not store `messages` otherwise."""

    text: str
    messages: list[dict]  # messages added by this run, in order
    status: str
    trace: Trace


def execute_tool_call(registry: ToolRegistry, name: str, raw_input: Any, ctx: Any, trace: Trace) -> ToolResult:
    """The one path every tool call takes, whether the model asked for it or the app resumes it."""
    started = time.perf_counter()
    result = registry.execute(name, raw_input, ctx)
    shown_input = {k: clip(v) for k, v in raw_input.items()} if isinstance(raw_input, dict) else clip(str(raw_input))
    trace.add(Span("tool", name, shown_input, clip(result.content), time.perf_counter() - started, is_error=result.is_error))
    return result


def _describe(message: dict) -> Any:
    content = message["content"]
    return clip(content) if isinstance(content, str) else f"{len(content)} blocks"


def _text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def run_agent(
    *, client: Any, model: str, system: str, registry: ToolRegistry, messages: list[dict],
    ctx: Any = None, max_tokens: int = 1024, max_steps: int = 8, trace: Trace | None = None,
) -> AgentResult:
    """max_steps counts model calls. At the limit, tools the model just asked for are NOT run."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    trace = trace if trace is not None else Trace()
    tools = registry.definitions()
    new: list[dict] = []
    for step in range(max_steps):
        history = messages + new
        started = time.perf_counter()
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system, tools=tools, messages=history,
        )
        in_tok = getattr(response.usage, "input_tokens", 0) or 0
        out_tok = getattr(response.usage, "output_tokens", 0) or 0
        trace.add(Span(
            "model", model,
            {"messages": len(history), "tools": [d["name"] for d in tools], "last": _describe(history[-1])},
            {"stop_reason": response.stop_reason, "text": clip(_text(response)),
             "tool_calls": [b.name for b in response.content if b.type == "tool_use"]},
            time.perf_counter() - started, in_tok, out_tok, trace.cost_usd(model, in_tok, out_tok),
        ))
        new.append({"role": "assistant", "content": response.content})
        reason = response.stop_reason
        if reason == "tool_use":
            if step == max_steps - 1:
                break
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    out = execute_tool_call(registry, block.name, block.input, ctx, trace)
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": out.content, "is_error": out.is_error,
                    })
            new.append({"role": "user", "content": results})
        elif reason != "pause_turn":  # pause_turn: send the turn back unchanged so the model can continue
            return AgentResult(_text(response), new, STATUS_FOR_STOP_REASON.get(reason, "unexpected_stop"), trace)
    return AgentResult(_text(response), new, "step_limit", trace)
